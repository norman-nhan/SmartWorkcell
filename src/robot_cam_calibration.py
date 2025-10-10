from ArucoDetection import ArucoDetectionNode
import os
import cv2.aruco as aruco
from RealSenseCamera import RealsenseCameraNode
from SmartWorkcell.calibration_utils import get_camera_intrinsic, rvec2matrix, make_transform_matrix, invert_transform, save_transform_matrix

CALIBRATION_PATH = "config/realsense_calibration_chessboard6x4-40mm.yaml" 
SAVE_DIR = "images/aruco/results"
IMAGE_DIR = "images/aruco/input"
TRANSFORM_PATH = "config/T_robot_cam.yaml"

def get_robot_to_cam_transform_matrix(T_cam_marker, T_marker_robot):
    # Compute camera to robot transform matrix
    T_cam_robot = T_cam_marker @ T_marker_robot
    # Compute robot to camera transform matrix
    T_robot_cam = invert_transform(T_cam_robot)

    # Save T_robot_cam
    save_transform_matrix(T_robot_cam, TRANSFORM_PATH)
    return T_robot_cam

if __name__ == "__main__":
    # Intialize camera node
    cam_node = RealsenseCameraNode(image_save_dir=IMAGE_DIR)
    # Intialize aruco detection node
    cam_mtx, dist_coeffs = get_camera_intrinsic(CALIBRATION_PATH)
    aruco_detection_node = ArucoDetectionNode(aruco.DICT_4X4_100, 0.1, cam_mtx, dist_coeffs, parameters=None, save_dir=SAVE_DIR)
    
    # Open camera, press 's' to save image
    cam_node.streaming()

    # Detect marker pose
    aruco_detection_node.detect_with_images(image_dir=IMAGE_DIR)