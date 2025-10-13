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
from ArucoDetection import ArucoDetectionNode

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