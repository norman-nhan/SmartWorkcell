#!/usr/bin/env python3
import rospy
import rs_aruco_detection
from pysmartworkcell import calibration_utils as calib_utils
import cv2.aruco as aruco
from geometry_msgs.msg import Pose

# Get package root path
pkg_root = calib_utils.find_pkg_path()

class VLMObjectNode:
    def __init__(self):
        rospy.init_node('VLM_object_node')
        self.pub = rospy.Publisher('object_pose', Pose, queue_size=10)
        rospy.loginfo('VLM_object_node initialized')
# ================================================================================== #
# === Compute Transformation matrix from robot coordinates to camera coordinates === #
# ================================================================================== #
print('[INFO] Starting calibrating camera and robot!')
input('======== Press `Enter` to start calibrating ========')

# 1. Detect marker and compute transformation matrix from camera to marker
success, cm_ids, T_cm_list = rs_aruco_detection.detect(aruco.DICT_4X4_50, marker_len=0.01, calib_path=pkg_root/'config'/'d435_origin.yaml')

# 2. Load transformation matrix from marker to robot (robot pose in marker coordinates)
mr_ids, T_mr_list = calib_utils.load_transform_mtx(pkg_root/'config'/'marker2robot.yaml')

# 3. Compute transformation matrix from robot to camera
T_rc_list = [calib_utils.invert_transform(T_cm @ T_mr) for T_cm, T_mr in zip(T_cm_list, T_mr_list)]
print('[INFO] Saved transformation matrix from robot coordinates to camera coordinates')

# ============================================= #
# === Detect object and estimate pose of it === #
# ============================================= #
CAPTION = input('========= Tell me what object do you need? =========')

def main():
    rospy.init_node('example_node')
    rospy.loginfo("Node started!")
    rospy.spin()

if __name__ == '__main__':
    main()
