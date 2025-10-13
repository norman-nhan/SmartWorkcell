import glob, os
from pathlib import Path
import argparse
import cv2
import cv2.aruco as aruco
import numpy as np
import time
from SmartWorkcell.calibration_utils import (
    get_camera_intrinsic, vectors2matrix,
    save_multi_transforms
)
import concurrent.futures

class ArucoDetectionNode:
    def __init__(self, dictionary: int, cam_matrix: np.ndarray, dist_coeffs: np.ndarray, parameters=None, marker_length: float=0.1,
                 save_dir="io/aruco/results"):
        
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
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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

    def estimate_marker_poses_from_frame(self, frame):
        """Estimate marker pose from a single frame.
        
        Returns
        -------
        - bool: marker found
        - ids: a list of detected marker ids
        - T_list: a list of transformation matrices"""
        marker_found = False
        T_list = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None:
            marker_found = True
            rvecs, tvecs = self.estimatePoseSingleMarkers(corners)
            for rvec, tvec in zip(rvecs, tvecs):
                axis_length = min(0.02, self.marker_length / 2.0)
                cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, rvec, tvec, axis_length)
                T = vectors2matrix(rvec=rvec, tvec=tvec)
                T_list.append(T)

        return marker_found, ids, T_list
    
    def _process_image(self, fname):
        """Handle each image in different thread"""
        img = cv2.imread(fname)
        success, ids, T_list = self.estimate_marker_poses_from_frame(img)

        # Save visualization
        img_path = os.path.join(self.save_dir, f"{Path(fname).stem}.png")
        cv2.imwrite(img_path, img)
        print(f'Saved image to {img_path}')
        save_path = os.path.join(self.save_dir, f'{Path(fname).stem}_pose.yaml')
        if success:
            save_multi_transforms(ids, T_list, save_path)
        return success, ids, T_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dictionary", type=str, default="DICT_4X4_50", help="Marker dictionary. DEFAULT: DICT_4X4_50")
    parser.add_argument("-l", "--marker_length", type=float, default=0.1, help="Marker length in meters. DEFAULT: 0.1")
    parser.add_argument("-i", "--image_dir", type=str, default='io/aruco/input', help='dir contains marker images')
    parser.add_argument("-o", "--save_dir", type=str, default="config", help="dir to save transformation matrices")
    parser.add_argument("-p", "--calibration_path", type=str, default="config/cam_calibration.yaml", help="This path contains camera matrix and dist_coeffs")
    args = parser.parse_args()

    # Load camera intrinsic
    cam_mtx, dist_coeffs = get_camera_intrinsic(args.calibration_path)

    node = ArucoDetectionNode(
        dictionary=getattr(aruco, args.dictionary),
        marker_length=args.marker_length, # in meters,
        cam_matrix=cam_mtx, dist_coeffs=dist_coeffs
    )

    # Gather image files (common extensions)
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    images = []
    for p in patterns:
        images.extend(glob.glob(os.path.join(args.image_dir, p)))
    images = sorted(images)
    if len(images) == 0:
        print(f"[ERROR] No images found in {args.image_dir}")
        return
    
    print(f'[INFO] Processing {len(images)}')
    start_time = time.time() # start time
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(node._process_image, fname) for fname in images]

        for f in concurrent.futures.as_completed(futures):
            success, ids, T_list = f.result()
    print(f"[INFO] Done in {time.time() - start_time:.2f} sec") # compute end time

if __name__ == "__main__":
    main()