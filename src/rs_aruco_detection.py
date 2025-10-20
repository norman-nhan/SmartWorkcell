import time
import numpy as np
from pysmartworkcell.ArucoDetection import ArucoDetectionNode
from pysmartworkcell import (
    calibration_utils as calib_utils,
    realsense_utils as rs_utils
)
import pyrealsense2 as rs
from pyrealsense2 import camera_info

def detect(dict: int, marker_len: float, calib_path: str):
    pkg_root = calib_utils.find_pkg_path()
    cam_mtx, dist_coeffs = calib_utils.load_camera_intrinsic(calib_path)
    detectNode = ArucoDetectionNode(
        dict, cam_matrix=cam_mtx, dist_coeffs=dist_coeffs, marker_length=marker_len
    )
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
            
            # Detect aruco marker
            success, ids, T_list = detectNode.estimate_maker_pose_from_frame(color_img)
            if success:
                print('Markers found! Compute transformation matrices...')
                save_path = pkg_root/'config'/'cam2marker.yaml'
                calib_utils.save_transform_mtx(ids, T_list=T_list, path=save_path)
                return success, ids, T_list

    except Exception as e:
        raise e
    finally:
        pipeline.stop()
        print('RealSense camera shutdown successfully!')
