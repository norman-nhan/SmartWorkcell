import os
import glob
import yaml
import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
from realsense_utils import RealsenseCameraNode
from pathlib import Path

class ArucoDetectionNode:
    def __init__(self, dictionary, marker_length, calibration_path=None, cam_matrix=None, dist_coeffs=None, parameters=None):
        self.dictionary = aruco.getPredefinedDictionary(dictionary)
        self.parameters = parameters if parameters is not None else aruco.DetectorParameters()
        # Tuning parameters
        self.parameters.minMarkerPerimeterRate = 0.05
        self.parameters.maxMarkerPerimeterRate = 3.0
        self.parameters.polygonalApproxAccuracyRate = 0.02
        self.parameters.minCornerDistanceRate = 0.1
        self.parameters.minMarkerDistanceRate = 0.05
        self.parameters.errorCorrectionRate = 0.3
        self.parameters.maxErroneousBitsInBorderRate = 0.02
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        if calibration_path is None and (cam_matrix is None or dist_coeffs is None):
            raise ValueError("[ERROR] Please provide camera intrinsic (either path or both matrix and coefficients).")
        if calibration_path is not None:
            self.load_calibration_data_from_file(calibration_path)
        else:
            self.cam_matrix = np.array(cam_matrix, dtype=np.float64)
            self.dist_coeffs =np.array(dist_coeffs, dtype=np.float64)
        
        self.marker_length = marker_length
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

    def load_calibration_data_from_file(self, path):
        path = Path(path)
        calibrated = None
        if path.suffix in [".yaml", ".yml"]: # check if yaml
            with open(path, 'r') as f:
                calibrated = yaml.safe_load(f)
        else: # other format like npy, npz
            calibrated = np.load(path)
        
        self.cam_matrix = np.array(calibrated["camera_matrix"], dtype=np.float64)
        self.dist_coeffs =np.array(calibrated["dist_coeffs"], dtype=np.float64)

    def estimatePoseSingleMarkers(self, corners):
        # Prepare 3D object points
        half_length = self.marker_length/2
        objp = np.array([
            [-half_length,  half_length, 0],
            [ half_length,  half_length, 0],
            [ half_length, -half_length, 0],
            [-half_length, -half_length, 0],
        ], dtype=np.float32)

        rvecs, tvecs = [], []
        for corner in corners:
            success, rvec, tvec = cv2.solvePnP(
                objp, corner[0], self.cam_matrix, self.dist_coeffs,
                # flags=aruco.CORNER_REFINE_SUBPIX
            )
            if success:
                # rvec, tvec = cv2.solvePnPRefineLM(objp, corner[0], self.cam_matrix, self.dist_coeffs, rvec, tvec)
                rvecs.append(rvec)
                tvecs.append(tvec)
        return rvecs, tvecs

    def estimate_marker_pose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs = self.estimatePoseSingleMarkers(corners)
        
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(
                    frame, self.cam_matrix, self.dist_coeffs, rvec, tvec, self.marker_length/2.0
                )
                # print(f"ID: {ids}  rvec: {rvec.ravel()}  tvec: {tvec.ravel()}")
                print(f"ID: {ids}  rvec: {np.degrees(rvec).ravel()}  tvec: {np.degrees(tvec).ravel()}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dictionary", type=str, default="DICT_4X4_50", help="DEFAULT: DICT_4X4_50")
    parser.add_argument("-l", "--marker_length", type=float, default=0.1, help="DEFAULT: 10cm. Aruco marker length in meters")
    parser.add_argument("-p", "--calibration_path", type=str, default="../config/realsense_origin.yaml")
    parser.add_argument("-rs", "--realsense", action="store_true", help="Use RealSense camera as streaming device.")
    parser.add_argument("-s", "--serial_number", type=int, help="This flag only works if -rs is True.")
    parser.add_argument("-i", "--image_dir", type=str, default="images/aruco/input", help="This flag only works if --no_camera is True")
    parser.add_argument("-o", "--save_dir", type=str, default="images/aruco/results", help="This flag only works if --no_camera is True")
    parser.add_argument("--no_camera", type=str, help="Use saved images as input")
    args = parser.parse_args()

    node = ArucoDetectionNode(
        dictionary=getattr(aruco, args.dictionary),
        marker_length=args.marker_length, # in meters
        calibration_path=args.calibration_path
    )
    # If saved images used as input
    if args.no_camera:
        if args.image_dir is not None:
            images = glob.glob(os.path.join(args.image_dir, "*"))
            if len(images) == 0:
                raise ValueError("[ERROR] Images dir is empty!")
            for fname in images:
                frame = cv2.imread(fname).copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = node.detector.detectMarkers(gray)
                print(f"detectMarkers: {ids}, {corners}")
                if ids is not None:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    rvecs, tvecs = node.estimatePoseSingleMarkers(corners)
                
                    for rvec, tvec in zip(rvecs, tvecs):
                        distance = np.linalg.norm(tvec)
                        axis_len = min(0.02, distance * 0.2)  # 20% of distance, max 5 cm
                        cv2.drawFrameAxes(
                            frame, node.cam_matrix, node.dist_coeffs, rvec, tvec, axis_len
                        )
                        cv2.imshow("aruco pose", frame)
                        print(f"ID: {ids}  rvec: {rvec.ravel()}  tvec: {tvec.ravel()}")
                cv2.waitKey(20000)
            cv2.destroyAllWindows()
        else:
            print("[ERROR] Please provide images or video!")
    # If RealSense camera used as input
    elif args.realsense:
        print("[INFO] Using RealSense camera")
        rscam = RealsenseCameraNode(serial_number=args.serial_number)
        rscam.streaming(callback=node.estimate_marker_pose)
    # If USB cam used as input
    else:
        cap = cv2.VideoCapture(1) # For MacOS, if 0 is not work, change to 1.
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to grab frame")
                    break
                
                node.estimate_marker_pose(frame)
                cv2.imshow("Webcam", frame)

                # Press 'q' to quit
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()