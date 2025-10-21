#!/usr/bin/env python3
import rospy
from pysmartworkcell import calibration_utils as calib_utils
import cv2.aruco as aruco
from geometry_msgs.msg import Pose
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
        self.dict = rospy.get_param('marker_dict')
        self.marker_length = rospy.get_param('marker_length')
        self.camera_info_topic = rospy.get_param('camera_info_topic')
        self.predefined_marker_path = rospy.get_param('predefined_marker_path')
        self.color_topic = rospy.get_param('color_topic')
        self.camera_link = rospy.get_param('camera_link')
        
        self.cam_mtx = None
        self.dist_coeffs = None
        self.bridge = CvBridge()
        self.latest_frame = None
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # Create publisher & subcriber
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, queue_size=10, callback=self.get_camera_intrinsics)
        self.color_sub = rospy.Subscriber(self.color_topic, Image, queue_size=10, callback=self.image_callback)
        self.image_pub = rospy.Publisher('detected_marker', Image, queue_size=10)
        # ======================================= #
        # === Load marker pose in robot frame === #
        # ======================================= #
        self.broadcast_marker_list = []
        self.predefined_marker_list = []

        predefined_ids, predefined_T_list = calib_utils.load_transform_mtx(self.predefined_marker_path)
        for id, T in zip(predefined_ids, predefined_T_list):
            t, quat = calib_utils.matrix2quat(T)
            self.predefined_marker_list.append(
                {
                    'id': id,
                    'translation': t,
                    'rotation': quat,
                    'parent': 'base_link',
                    'child': f'marker_{id}_link'
                }
            )
                
        self.wait_for_camera_intrinsics()
        self.arucoDetection = ArucoDetection(
            dictionary=getattr(aruco, self.dict),
            marker_length=self.marker_length,
            cam_matrix=self.cam_mtx,
            dist_coeffs=self.dist_coeffs,
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
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_frame = cv_image
        except CvBridgeError as e:
            rospy.logerr(f'Failed to convert image due to {e}')
    
    def get_camera_intrinsics(self, camera_info) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.cam_mtx is None or self.dist_coeffs is None:
            self.cam_mtx = np.array(camera_info.K).reshape(3, 3)
            self.dist_coeffs = np.array(camera_info.D)
        return self.cam_mtx, self.dist_coeffs
        
    def detect_marker(self) -> Optional[Tuple[bool, List[int], List[np.ndarray]]]:
        image = self.latest_frame.copy()
        success, found_ids, found_T_list = self.arucoDetection.estimate_maker_pose_from_frame(image)
        rospy.loginfo_throttle(5.0, f"Founded marker are: {found_ids}")
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image))
        calibrated_ids, calibrated_marker_list = [], []
        
        if success:
            predefined_ids = [marker['id'] for marker in self.predefined_marker_list]
            for id, T in zip(found_ids, found_T_list):
                if id in predefined_ids:
                    calibrated_ids.append(id)
                    T_marker_cam = calib_utils.invert_transform(T)
                    t, quat = calib_utils.matrix2quat(T_marker_cam)
                    calibrated_marker_list.append(
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
        
        for pre_marker in self.predefined_marker_list:
            if pre_marker['id'] in calibrated_ids:
                self.broadcast_marker_list.append(pre_marker)
        self.broadcast_marker_list.extend(calibrated_marker_list)

        return success, found_ids, found_T_list

if __name__ == '__main__':
    node = CameraCalibrationNode()
    rospy.sleep(2)
    rate = rospy.Rate(10)
    rospy.loginfo("Publishing TF")
    try:
        while not rospy.is_shutdown(): # Keep publishing marker pose if marker found
            success, _, _ = node.detect_marker() # Detect marker
            if success:
                for marker in node.broadcast_marker_list:
                    node.tf_broadcaster.sendTransform(
                        marker['translation'],
                        marker['rotation'],
                        rospy.Time.now(),
                        marker['child'],
                        marker['parent']
                    )
            else:
                rospy.logwarn("No markers found. Skipping publishing TF...") # Shutdown node if no marker was found.
            rate.sleep()
    except rospy.ROSInterruptException:
        pass