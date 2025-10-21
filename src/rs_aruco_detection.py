import time
import numpy as np
from pysmartworkcell.ArucoDetection import ArucoDetection
from pysmartworkcell import (
    calibration_utils as calib_utils,
    realsense_utils as rs_utils
)
import pyrealsense2 as rs
from pyrealsense2 import camera_info
import cv2

def detect(dict: int, marker_len: float, calib_path: str):
    pkg_root = calib_utils.find_pkg_path()
    # cam_mtx, dist_coeffs = calib_utils.load_camera_intrinsic(calib_path)
    # detectNode = ArucoDetection(
    #     dict, cam_matrix=cam_mtx, dist_coeffs=dist_coeffs, marker_length=marker_len
    # )
    pipeline, config = rs_utils.initialize_realsense_camera()
    colorizer = rs.colorizer()
    try:
        pipeline.start(config)
        time.sleep(2)
        profile = pipeline.get_active_profile()
        device = profile.get_device()
        print(f'Streaming device: {device.get_info(camera_info.name)}\
            serial number: {device.get_info(camera_info.serial_number)}')
        while profile is not None:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            
            # Convert frame to np arrays
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            depth16 = np.asanyarray(depth_frame.get_data())

            image_concat = np.hstack((color_img, depth_img))
            cv2.imshow('rgb+depth', image_concat)
            
            # Save image and depth
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite('../io/color.png', color_img)
                np.save('../io/depth.npy', depth16)
                print(f'Image & Depth are saved.')
            if key == ord('q'):
                break
            
            # Detect aruco marker
            # success, ids, T_list = detectNode.estimate_maker_pose_from_frame(color_img)
            # if success:
            #     print('Markers found! Compute transformation matrices...')
            #     save_path = pkg_root/'config'/'cam2marker.yaml'
            #     calib_utils.save_transform_mtx(ids, T_list=T_list, path=save_path)
            #     return success, ids, T_list
      
    except Exception as e:
        raise e
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print('RealSense camera shutdown successfully!')

if __name__ == '__main__':
    detect(cv2.aruco.DICT_4X4_50, 0.01, '')