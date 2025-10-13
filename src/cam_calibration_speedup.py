import concurrent.futures
from SmartWorkcell.calibration_utils import save_camera_calibration
import os
import cv2
import numpy as np
import glob
import yaml
from pathlib import Path
import argparse

class ChessboardCalibration():
    def __init__(self, dims, size:float=0.04, image_dir='io/calibration/input', save_dir='io/calibration/results'):
        if dims is None:
            raise ValueError("Please provide chessboard size with flag: `python3 calibration.py -d COLS ROWS`")
        self.dims = dims # Chessboard inner corners (cols, rows)
        self.size = size if size is not None else 0.04 # chessboard square size in meters
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Criteria for corner refinement
        self.image_dir = image_dir if image_dir is not None else 'io/calibration/input'
        self.save_dir = save_dir if save_dir is not None else 'io/calibration/input'

        # Ensure save dir is exist before saving        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f'[INFO] Created directory: {self.save_dir}')
        
        # Prepare 3D objpoints
        cols, rows = self.dims
        print(f"[INFO] Calibrating with Chessboard-{cols}X{rows}-{self.size*1000}mm")
        self.empty_objp = np.zeros((rows * cols, 3), np.float32) # (H*W, 3)
        objp = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32) # make 2d grid
        self.empty_objp[:, :2] = objp * self.size # same z coordinate for every pixel

    def _process_image(self, fname):
        """Process one image and return (success, imgpoints(refined_corners), imageSize)"""
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(
            gray, self.dims,
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        print(f"[INFO] {fname} - pattern found: {success}")
        if not success:
            return None
        
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        img_path = os.path.join(self.save_dir, fname)
        cv2.drawChessboardCorners(img, self.dims, refined_corners, success)
        cv2.imwrite(img_path, img)
        return (success, refined_corners, tuple(gray.shape[::-1]))


    def calibrate(self):
        objpoints = []  # 3D points
        imgpoints = []  # 2D points
        used_imageSize = None

        # Gather image files
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        images = []
        for p in patterns:
            images.extend(glob.glob(os.path.join(self.image_dir, p)))
        images = sorted(images)
        if len(images) == 0:
            print(f"[ERROR] No images found in {self.image_dir}")
            return
        else:
            print(f"[INFO] Total images: {len(images)}")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(self._process_image, images))
        for r in results:
            if r is None:
                continue
            
            success, imgpoint, imageSize = r
            if used_imageSize is None:
                used_imageSize = imageSize
            elif used_imageSize != imageSize:
                print(f'[WARN] Skipping due to mismatched size\n Current size:{used_imageSize} != New size:{imageSize}')
                continue
            objpoints.append(self.empty_objp.copy())
            imgpoints.append(imgpoint)
        
        if not objpoints:
            print('[ERROR] No chessboard was found.')
            return

        # Camera calibration
        rms, cam_mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, used_imageSize, None, None
        )
        print(f"[INFO] RMS: {rms}")
        print("[INFO] Camera Matrix:\n", cam_mtx)
        print("[INFO] Distortion Coefficients:\n", dist_coeffs)

        # Compute reprojection error
        total_error = 0
        errors = []
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cam_mtx, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            errors.append(float(error))
            total_error += error
        mean_error = total_error / len(objpoints)
        print(f"[INFO] Reprojection error per image (mean): {mean_error}")
        print(f"[INFO] Reprojection errors: {errors}")

        # Save
        calibration_path = os.path.join(self.save_dir, "cam_calibration.yaml")
        save_camera_calibration(calibration_path, cam_mtx, dist_coeffs, rms, mean_error, errors)
        print(f"Saved camera calibration to {calibration_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dims", type=int, nargs=2, metavar=("COLS", "ROWS"),
                        help="number of inner corners per chessboard row and column (cols rows).")
    parser.add_argument("-s", "--size", type=float,
                        help="size of a square on the chessboard in meters.")
    parser.add_argument("-i", "--image_dir", type=str, default="io/calibration/input")
    parser.add_argument("-o", "--save_dir", type=str, default="io/calibration/results")
    args = parser.parse_args()
    
    calibration_node = ChessboardCalibration(dims=args.dims,
                                             size=args.size,
                                             image_dir=args.image_dir,
                                             save_dir=args.save_dir)
    calibration_node.calibrate()