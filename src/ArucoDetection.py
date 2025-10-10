import time
import os
import glob
import yaml
import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
from RealSenseCamera import RealsenseCameraNode
from pathlib import Path
from SmartWorkcell.calibration_utils import rvec2matrix, make_transform_matrix, save_transform_matrix, get_camera_intrinsic

class ArucoDetectionNode:
    def __init__(self, dictionary: int, marker_length: float, cam_matrix: np.ndarray, dist_coeffs: np.ndarray, parameters=None,
                 save_dir="images/aruco/results"):
        
        # Tuning parameters
        self.parameters = parameters if parameters is not None else aruco.DetectorParameters()
        self.parameters.minMarkerPerimeterRate = 0.05
        self.parameters.maxMarkerPerimeterRate = 3.0
        self.parameters.polygonalApproxAccuracyRate = 0.02
        self.parameters.minCornerDistanceRate = 0.1
        self.parameters.minMarkerDistanceRate = 0.05
        self.parameters.errorCorrectionRate = 0.3
        self.parameters.maxErroneousBitsInBorderRate = 0.02
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.dictionary = aruco.getPredefinedDictionary(dictionary)
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)
        self.cam_matrix = cam_matrix 
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length

        # Save detection result
        self.save_dir = save_dir


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
            )
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
        return rvecs, tvecs        

    def estimate_marker_pose(self, frame, is_streaming=True, fname=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        # print(f'[INFO] Detected markers: {ids}')
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs = self.estimatePoseSingleMarkers(corners)
            for id, rvec, tvec in zip(ids, rvecs, tvecs):
                distance = np.linalg.norm(tvec)
                axis_len = min(0.02, distance * 0.2) # to avoid axis length overflows
                cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, rvec, tvec, axis_len)
                if not is_streaming:
                    cv2.imshow("Marker pose", frame)
                print(f"[INFO] ID: {id}, rvec: {rvec}, tvec: {tvec}")
        
                # ================
                # Save marker pose
                # ================
                fname = Path(fname).stem if fname is not None else f'{id}_pose_at_{int(time.time() * 1000)}'
                save_path = os.path.join(self.save_dir, f"{fname}_pose.yaml")
                if not is_streaming:
                    R = rvec2matrix(rvec)
                    # Make tranform matrix from camera to marker
                    T = make_transform_matrix(R, tvec)
                    save_transform_matrix(T, save_path)
                if is_streaming:
                    key = cv2.waitKey(1) 
                    if key == ord('s'):
                        print("[INFO] Computing camera to marker transform matrix!")
                        img_save_path = os.path.join(self.save_dir, f'{fname}.png')
                        cv2.imwrite(img_save_path, frame)
                        print(f"Save image to {img_save_path}")
                        R = rvec2matrix(rvec)
                        # Make tranform matrix from camera to marker
                        T = make_transform_matrix(R, tvec)
                        save_transform_matrix(T, save_path)
                        time.sleep(0.5) # wait to save transform matrix before loading next frame
        if not is_streaming:        
            cv2.destroyAllWindows()

    def detect_with_realsense(self, serial_number=None):
        print("[INFO] USING REALSENSE MODE!")
        rscam = RealsenseCameraNode(serial_number=serial_number, image_save_dir=self.save_dir, save_depth=False)
        rscam.streaming(self.estimate_marker_pose)
        
    def detect_with_images(self, image_dir="images/aruco/input"):
        print("[INFO] USING IMAGE MODE!")
        patterns = [".png", ".jpeg", ".jpg", ".tiff"]
        images = []
        for _ in patterns:
            images.extend(glob.glob(os.path.join(image_dir, "*.png")))
        images = sorted(images)
        if len(images) == 0:
            print(f"[ERROR] No images found in {image_dir}")
            return
        
        for fname in images:
            frame = cv2.imread(fname)
            self.estimate_marker_pose(frame, fname, is_streaming=False)

    def detect_with_webcam(self, camera_id=1):
        cap = cv2.VideoCapture(camera_id)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to grab frame")
                    break
                self.estimate_marker_pose(frame, is_streaming=True)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dictionary", type=str, default="DICT_4X4_50", help="DEFAULT: DICT_4X4_50")
    parser.add_argument("-l", "--marker_length", type=float, default=0.1, help="DEFAULT: 10cm. Aruco marker length in meters")
    parser.add_argument("-rs", "--realsense", action="store_true", help="Use RealSense camera as streaming device.")
    parser.add_argument("-nc", "--no_camera", action="store_true", help="Use saved images as input")
    parser.add_argument("-s", "--serial_number", type=int, help="This flag only works if -rs is True.")
    parser.add_argument("-i", "--image_dir", type=str, default="images/aruco/input", help="This flag only works if --no_camera is True")
    parser.add_argument("-o", "--save_dir", type=str, default="images/aruco/results", help="images and yaml save dir")
    parser.add_argument("-p", "--calibration_path", type=str, default="config/realsense_origin.yaml", help="This path contains camera matrix and dist_coeffs")
    args = parser.parse_args()

    # Load camera intrinsic
    cam_mtx, dist_coeffs = get_camera_intrinsic(args.calibration_path)

    node = ArucoDetectionNode(
        dictionary=getattr(aruco, args.dictionary),
        marker_length=args.marker_length, # in meters,
        cam_matrix=cam_mtx, dist_coeffs=dist_coeffs
    )
    # If saved images used as input
    if args.no_camera:
        node.detect_with_images(args.image_dir)
    # If RealSense camera used as input
    elif args.realsense:
        node.detect_with_realsense(args.serial_number)
    # If USB cam used as input
    else:
        node.detect_with_webcam(1)

if __name__ == "__main__":
    main()