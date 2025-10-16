from SmartWorkcell.ArucoDetection import ArucoDetectionNode
import cv2.aruco as aruco
import cv2
from SmartWorkcell.calibration_utils import (
    load_camera_intrinsic, load_multi_transforms, save_multi_transforms,
    invert_transform
)
from SmartWorkcell.realsense_utils import initialize_realsense_camera, print_available_sensors
import pyrealsense2 as rs
import numpy as np
from pathlib import Path
import time

def main():
    # Load transformation matrix from marker -> robot
    marker_robot_ids, marker_robot_T_list = load_multi_transforms('config/marker2robot.yaml')
    cam_mtx, dist_coeffs = load_camera_intrinsic('config/realsense_origin.yaml')
    detectionNode = ArucoDetectionNode(
        aruco.DICT_4X4_50,
        cam_mtx, dist_coeffs,
    )
    # Initialize realsense camera
    pipeline, config = initialize_realsense_camera()
    colorizer = rs.colorizer()
    try:
        pipeline.start(config)
        time.sleep(2)
        profile = pipeline.get_active_profile()
        device = profile.get_device()
        # print_available_sensors(device)
        print(f"[INFO] Streaming with device {device.get_info(rs.camera_info.name)} (serial number: {device.get_info(rs.camera_info.serial_number)})")
        while profile is not None:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # Convert rs's frame to np
            color_frame = np.asanyarray(color_frame.get_data())
            colored_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            depth_frame = np.asanyarray(depth_frame.get_data())
            
            # Detect markers
            success, ids, T_list = detectionNode.estimate_marker_poses_from_frame(color_frame)
            robot_cam_T_list = []
            if success:
                for id, T_cam_marker in zip(ids, T_list):
                    if id[0] in marker_robot_ids:
                        T_cam_robot = T_cam_marker @ marker_robot_T_list[id[0]]
                        robot_cam_T_list.append(invert_transform(T_cam_robot))
                save_multi_transforms(ids, robot_cam_T_list, Path(__file__).parent.parent/'config'/'robot2cam.yaml')
                break

            # Show frame in opencv
            concat_frame = np.hstack((color_frame, colored_depth))
            cv2.imshow("RGB + Depth", concat_frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    except Exception as e:
        print(f'{e}')
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print('[INFO] Device shutdown successfully!')


def test_cam():
    pipeline, config = initialize_realsense_camera()
    colorizer = rs.colorizer()
    pipeline.start(config)
    time.sleep(2) # wait for pipeline completely started
    try:
        profile = pipeline.get_active_profile()
        device = profile.get_device()
        # print_available_sensors(device)
        print(f"[INFO] Streaming with device {device.get_info(rs.camera_info.name)} (serial number: {device.get_info(rs.camera_info.serial_number)})")
        while profile is not None:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # Convert rs's frame to np
            color_frame = np.asanyarray(color_frame.get_data())
            colored_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            depth_frame = np.asanyarray(depth_frame.get_data())
            # Show frame in opencv
            concat_frame = np.hstack((color_frame, colored_depth))
            cv2.imshow("RGB + Depth", concat_frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    except Exception as e:
        print(f'{e}')
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print('[INFO] Device shutdown successfully!')
if __name__ == "__main__":
    main()
    # test_cam()