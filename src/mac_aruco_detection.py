import time
import yaml
import cv2
import cv2.aruco as aruco
import numpy as np
import os
from SmartWorkcell.calibration_utils import rvec2matrix, make_transform_matrix, save_transform_matrix

SAVE_DIR = "images/aruco/results"
CALIB_PATH = "config/mac_camera_calibration.yaml"
MARKER_LENGTH = 0.05 # in meters 

def estimatePoseSingleMarkers(corners, cam_matrix, dist_coeffs):
    # Prepare 3D object points
    half_length = MARKER_LENGTH/2
    objp = np.array([
        [-half_length,  half_length, 0],
        [ half_length,  half_length, 0],
        [ half_length, -half_length, 0],
        [-half_length, -half_length, 0],
    ], dtype=np.float32)

    rvecs, tvecs = [], []
    for corner in corners:
        success, rvec, tvec = cv2.solvePnP(
            objp, corner[0], cam_matrix, dist_coeffs,
        )
        if success:
            rvecs.append(rvec)
            tvecs.append(tvec)
    return rvecs, tvecs


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f'[INFO] Created directory {SAVE_DIR}')
    
    # Get marker dictionary
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    # Tuning parameters
    params = aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.05
    params.maxMarkerPerimeterRate = 3.0
    params.polygonalApproxAccuracyRate = 0.02
    params.minCornerDistanceRate = 0.1
    params.minMarkerDistanceRate = 0.05
    params.errorCorrectionRate = 0.3
    params.maxErroneousBitsInBorderRate = 0.02
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    # Init detector
    detector = aruco.ArucoDetector(dictionary, params)
    with open(CALIB_PATH, 'r') as f:
        calibrated = yaml.safe_load(f)
    cam_matrix = np.array(calibrated["camera_matrix"], dtype=np.float64)
    dist_coeffs =np.array(calibrated["dist_coeffs"], dtype=np.float64)
    cap = cv2.VideoCapture(1)
    print("Initialization completed.")
    try:
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Unable to receive frame")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs = estimatePoseSingleMarkers(corners, cam_matrix, dist_coeffs)
                for id, rvec, tvec in zip(ids, rvecs, tvecs):
                    cv2.drawFrameAxes(
                        frame, cam_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH/2.0
                    )
                    # print(f"ID: {id}, rvec: {rvec}, tvec: {tvec}")
            
            # Show frame
            cv2.imshow('aruco marker pose', frame)

            # Handle key signals
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('s'):
                img_save_path = os.path.join(SAVE_DIR, f"{count}.png")
                cv2.imwrite(img_save_path, frame)
                print(f"[INFO] Saved to {img_save_path}.")
                yaml_save_path = os.path.join(SAVE_DIR, f'{count}_pose.yaml')
                if ids is not None:
                    R = rvec2matrix(rvec)
                    T = make_transform_matrix(R, tvec)
                    save_transform_matrix(T, yaml_save_path)
                    time.sleep(1) # wait to compute and save transform matrix
                count+=1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] All processes shutdown successfully!")

if __name__ == "__main__":
    main()