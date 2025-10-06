import os
import cv2
import numpy as np
import glob
import yaml
from pathlib import Path
import argparse

class ChessboardCalibration():
    def __init__(self, chessboard_size, image_dir, save_dir):
        self.chessboard_size = chessboard_size
        self.image_dir = image_dir
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f'[INFO] Created directory: {self.save_dir}')
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Criteria for corner refinement

        # Prepare 3D points in real-world space (z=0 plane)
        n_cols, n_rows = self.chessboard_size
        self.empty_objp = np.zeros((n_rows * n_cols, 3), np.float32)
        self.empty_objp[:, :2] = np.mgrid[0:n_rows, 0:n_cols].T.reshape(-1, 2) # z-coord is at 0

    def save_calib_data(self, camera_matrix, dist_coeffs, overall_rms):
        np_path = os.path.join(os.getcwd(), "camera_calibration.npz")
        np.savez(np_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, overall_rms=overall_rms)
        
        calibration_data = {
            'camera_matrix': camera_matrix.tolist(), # 3x3
            'dist_coeffs': dist_coeffs.flatten().tolist(), # 1x5
            'overall_rms': overall_rms
        }
        yaml_path = os.path.join(os.getcwd(), "camera_calibration.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(
                calibration_data,
                f,
                sort_keys=False,
                default_flow_style=None,  # auto: list stays inline if short
                width=120,                # keeps long lines from wrapping
                indent=2                  # clean indentation for readability
            )
            print(f'[INFO] Saved calibration data to {yaml_path}')

    def calibrate(self):
        objpoints = []  # 3D points
        imgpoints = []  # 2D points

        # Load all calibration images (change path)
        images = glob.glob(os.path.join(self.image_dir, "*"))
        if len(images) == 0: # check if image dir is empty
            print(f"[ERROR] No images found in {self.image_dir}")
            return # stop calibration

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            patternfound, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None, 
                                                     flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                                                        + cv2.CALIB_CB_FAST_CHECK 
                                                        + cv2.CALIB_CB_NORMALIZE_IMAGE)
            print(f"[DEBUG] corners found: {corners}")

            if patternfound:
                objpoints.append(self.empty_objp)

                # Refine corner locations
                # (11, 11) is the searching kernel size
                # (-1, -1) to disable zerozone
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(refined_corners)

                # Draw & show
                cv2.drawChessboardCorners(img, self.chessboard_size, refined_corners, patternfound)
                cv2.imshow('Found chessboard', img)
                cv2.waitKey(500)
                fname = Path(fname).stem
                save_path = os.path.join(self.save_dir, f'found_chessboard_in_{fname}.png')
                cv2.imwrite(save_path, img)
                print(f'[INFO] Saved found chessboard to {save_path}')

        cv2.destroyAllWindows()

        # Camera calibration
        overall_rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        print("[INFO] Camera Matrix:\n", camera_matrix)
        print("[INFO] Distortion Coefficients:\n", dist_coeffs)
        print(f"[INFO] Overall RMS: {overall_rms}")

        # Save to file
        self.save_calib_data(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, overall_rms=overall_rms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--chessboard_size", type=int, nargs=2)
    parser.add_argument("-i", "--image_dir", type=str, default="images/calibration/input")
    parser.add_argument("-o", "--save_dir", type=str, default="images/calibration/results")
    args = parser.parse_args()
    
    calibration_node = ChessboardCalibration(chessboard_size=args.chessboard_size, image_dir=args.image_dir, save_dir=args.save_dir)
    calibration_node.calibrate()