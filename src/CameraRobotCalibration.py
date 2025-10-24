#!/usr/bin/env python3
import rospy
from pysmartworkcell import calibration_utils as calib_utils
import cv2.aruco as aruco
from sensor_msgs.msg import CameraInfo, Image
import numpy as np
from typing import Optional, Tuple, List
from pysmartworkcell.ArucoDetection import ArucoDetection
from cv_bridge import CvBridge, CvBridgeError
import tf

class CameraCalibrationNode():
    def __init__(self):
        rospy.init_node('camera_calibration_node')
        
        # Get node params
        self.dict              = rospy.get_param('~marker_dict', 'DICT_4X4_50')
        self.marker_length     = rospy.get_param('~marker_length', 0.03)
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/aligned_depth_to_color/camera_info')
        self.color_topic       = rospy.get_param('color_topic', '/camera/color/image_raw')
        self.camera_link       = rospy.get_param('camera_link', 'camera_link')
        
        pkg_root = calib_utils.find_pkg_path()
        default_calib_path = pkg_root / 'config' / 'calibrated_marker.yaml'
        self.calibrated_marker_path = rospy.get_param('~calibrated_marker_path', default_calib_path)
        
        self.cam_mtx = None
        self.dist_coeffs = None
        self.latest_frame = None
        self.bridge = CvBridge()
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # Create publisher & subcriber
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, queue_size=10, callback=self.get_camera_intrinsics)
        self.color_sub       = rospy.Subscriber(self.color_topic, Image, queue_size=10, callback=self.image_callback)
        self.image_pub       = rospy.Publisher('detected_marker', Image, queue_size=10)
        
        # ============================================= #
        # === LOAD MARKER POSE IN ROBOT COORDINATES === #
        # ============================================= #
        self.broadcast_marker_list = []
        self.calibrated_marker_list = []

        calibrated_ids, calibrated_T_list = calib_utils.load_transform_mtx(self.calibrated_marker_path)
        for id, T in zip(calibrated_ids, calibrated_T_list):
            # Since I save robot pose in marker coordinates so I need to convert it to have robot -> marker TF
            robot_to_marker_T = calib_utils.invert_transform(T)
            t, quat = calib_utils.matrix2quat(robot_to_marker_T)
            self.calibrated_marker_list.append(
                {
                    'id': id,
                    'translation': t,
                    'rotation': quat,
                    'parent': 'base_link',
                    'child': f'marker_{id}_link'
                }
            )
        
        # Initialize Aruco Marker Detection
        self.wait_for_camera_intrinsics()
        self.arucoDetection = ArucoDetection(
            dictionary    = getattr(aruco, self.dict),
            marker_length = self.marker_length,
            cam_matrix    = self.cam_mtx,
            dist_coeffs   = self.dist_coeffs,
        )
        
        rospy.loginfo("=== camera_calibration_node initialization complete! ===")

    def wait_for_camera_intrinsics(self, timeout = 10.0):
        if self.cam_mtx is None or self.dist_coeffs is None:
            rospy.loginfo('Wait for camera intrinsics...')
            start = rospy.Time.now().to_sec()
            while (self.cam_mtx is None or self.dist_coeffs is None) and (rospy.Time.now().to_sec() - start) < timeout:
                rospy.sleep(0.1)
            if self.cam_mtx is None or self.dist_coeffs is None:
                raise RuntimeError('[Timeout] Cannot receive camera intrinsics. Did you run camera node?')
            rospy.loginfo("Received camera intrinsics!")

    def image_callback(self, msg):
        """Convert ros msg to cv img and assign to self.latest_frame"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_frame = cv_image
        except CvBridgeError as e:
            rospy.logerr(f'Failed to convert image due to {e}')
    
    def get_camera_intrinsics(self, camera_info) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Read camera intrinsics from CameraInfo topic"""
        if self.cam_mtx is None or self.dist_coeffs is None:
            self.cam_mtx = np.array(camera_info.K).reshape(3, 3)
            self.dist_coeffs = np.array(camera_info.D)
        return self.cam_mtx, self.dist_coeffs
        
    def detect_marker(self) -> Tuple[bool, Optional[List[int]], Optional[List[np.ndarray]]]:
        """Detect marker, Return list(int) of marker id and list(np.ndarray) of transform matrix marker->cam.
        
        It also update self.broadcast_marker_list!
        """
        image = self.latest_frame.copy()
        success, found_ids, found_T_list = self.arucoDetection.estimate_maker_pose_from_frame(image)
        rospy.loginfo_throttle(5.0, f"Founded marker are: {found_ids}")
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image))
        
        matched_ids, matched_marker_list = [], [] # Variables to get marker that matched in pre-calibrated list
        if success:
            calib_ids = [marker['id'] for marker in self.calibrated_marker_list]
            for id, T in zip(found_ids, found_T_list):
                if id in calib_ids:
                    matched_ids.append(id)
                    # Since detected marker pose are in camera frame, we need to invert it to have marker -> cam transform matrix
                    marker_to_cam_T = calib_utils.invert_transform(T)
                    t, quat = calib_utils.matrix2quat(marker_to_cam_T)
                    matched_marker_list.append(
                        {
                            'id': id,
                            'translation': t,
                            'rotation': quat,
                            'child': self.camera_link,
                            'parent': f'marker_{id}_link'
                        }
                    )
        else:
            rospy.logwarn_throttle(5.0, 'No markers are found...')
            return False, [], []
        
        # append marker to broadcast list
        self.broadcast_marker_list = [] # reset broadcast list every time detect_marker is called
        
        for marker in self.calibrated_marker_list:
            if marker['id'] in matched_ids:
                self.broadcast_marker_list.append(marker) # add only marker that in matched list
        self.broadcast_marker_list.extend(matched_marker_list) # extend all found matched markers

        return success, found_ids, found_T_list

if __name__ == '__main__':
    node = CameraCalibrationNode()
    rospy.sleep(2) # Wait to initialization completed
    rate = rospy.Rate(10) # Set rate
    try:
        while not rospy.is_shutdown(): # Keep publishing marker pose if marker found
            success, _, _ = node.detect_marker() # Detect marker and update broadcast marker list
            if success:
                rospy.loginfo_throttle("Publishing TF")
                for marker in node.broadcast_marker_list:
                    node.tf_broadcaster.sendTransform(
                        marker['translation'],
                        marker['rotation'],
                        rospy.Time.now(),
                        marker['child'],
                        marker['parent']
                    )
            else:
                rospy.logwarn("No markers found. Skipping publishing TF...")
            rate.sleep()
    except rospy.ROSInterruptException:
        pass