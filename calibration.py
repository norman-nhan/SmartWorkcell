import os
import cv2
import numpy as np
import glob
import yaml
from pathlib import Path
import argparse

class ChessboardCalibration():
    def __init__(self, chessboard_size, square_size, image_dir, save_dir, show=False):
        if chessboard_size is None:
            raise ValueError("Please provide [cols rows] (number of chessboard inner corners)")
        if square_size is None:
            raise ValueError("Please provide chessboard square size in meters.")
        
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.image_dir = image_dir
        self.show = show
        
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f'[INFO] Created directory: {self.save_dir}')
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Criteria for corner refinement
        
        # Prepare 3D points in real-world space (z=0 plane).
        cols, rows = self.chessboard_size
        print(f'[INFO] chessboard_size interpreted as (cols, rows) = ({cols}, {rows})')
        self.empty_objp = np.zeros((rows * cols, 3), np.float32)
        objp = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
        self.empty_objp[:, :2] = objp * self.square_size

        print(f"[INFO] Using Chessboard-{cols}X{rows}-{square_size*1000}mm")

    def save2yaml(self, camera_matrix, dist_coeffs, overall_rms):
        calibration_data = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.flatten().tolist() if hasattr(dist_coeffs, 'flatten') else list(dist_coeffs),
            'overall_rms': float(overall_rms)
        }
        yaml_path = os.path.join(self.save_dir, "camera_calibration.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(
                calibration_data,
                f,
                sort_keys=False,
                default_flow_style=None,
                width=120,
                indent=2
            )
        print(f'[INFO] Saved calibration data to {yaml_path}')

    def calibrate(self):
        objpoints = []  # 3D points
        imgpoints = []  # 2D points

        # Gather image files (common extensions)
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        images = []
        for p in patterns:
            images.extend(glob.glob(os.path.join(self.image_dir, p)))
        images = sorted(images)
        if len(images) == 0:
            print(f"[ERROR] No images found in {self.image_dir}")
            return

        used_image_shape = None
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                print(f"[WARN] Could not read {fname}, skipping")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Save image shape for calibrateCamera (must be consistent)
            if used_image_shape is None:
                used_image_shape = gray.shape[::-1]
            elif used_image_shape != gray.shape[::-1]:
                print(f"[WARN] Image {fname} has different size {gray.shape[::-1]} vs {used_image_shape}, skipping")
                continue

            # Find chessboard corners. Remove FAST_CHECK for robustness during calibration.
            patternfound, corners = cv2.findChessboardCorners(
                gray, self.chessboard_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            print(f"[INFO] {fname} - pattern found: {patternfound}")

            if patternfound:
                # store a copy of object points for each image
                objpoints.append(self.empty_objp.copy())

                # Refine corner locations
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(refined_corners)

                # Draw & optionally show
                cv2.drawChessboardCorners(img, self.chessboard_size, refined_corners, patternfound)
                if self.show:
                    cv2.imshow('Found chessboard', img)
                    cv2.waitKey(500)
                stem = Path(fname).stem
                save_path = os.path.join(self.save_dir, f'found_chessboard_in_{stem}.png')
                cv2.imwrite(save_path, img)
                print(f'[INFO] Saved found chessboard to {save_path}')

        if self.show:
            cv2.destroyAllWindows()

        if len(objpoints) == 0:
            print('[ERROR] No chessboard patterns were found. Check images and chessboard_size.')
            return

        # Camera calibration
        overall_rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, used_image_shape, None, None
        )

        print("[INFO] Camera Matrix:\n", camera_matrix)
        print("[INFO] Distortion Coefficients:\n", dist_coeffs)
        print(f"[INFO] Overall RMS: {overall_rms}")

        # Compute reprojection error
        total_error = 0
        errors = []
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            errors.append(float(error))
            total_error += error
        mean_error = total_error / len(objpoints)
        print(f"[INFO] Reprojection error per image (mean): {mean_error}")
        print(f"[INFO] Reprojection errors: {errors}")

        # Save to file
        self.save2yaml(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, overall_rms=overall_rms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chessboard_size", type=int, nargs=2, metavar=("COLS", "ROWS"),
                        help="number of inner corners per chessboard row and column (cols rows).")
    parser.add_argument("--square_size", type=float,
                        help="size of a square on the chessboard in chosen units (e.g. meters).")
    parser.add_argument("-i", "--image_dir", type=str, default="images/calibration/input")
    parser.add_argument("-o", "--save_dir", type=str, default="images/calibration/results")
    parser.add_argument("--show", action='store_true', help="show detected corners during processing")
    args = parser.parse_args()
    
    calibration_node = ChessboardCalibration(chessboard_size=args.chessboard_size,
                                             image_dir=args.image_dir,
                                             save_dir=args.save_dir,
                                             square_size=args.square_size,
                                             show=args.show)
    calibration_node.calibrate()