#!/usr/bin/env python3

# This node should detect obj but only when we give the command
from typing import Optional, Tuple
import PIL
import cv2
import numpy as np
import supervision as sv
import message_filters
from pysmartworkcell import (
    vlm_utils, 
    calibration_utils as calib_utils
)
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from smartworkcell.srv import DetectObjectPose, DetectObjectPoseResponse, DetectObjectPoseRequest

class ObjectDetectNode():
    def __init__(self):
        rospy.init_node('object_detect_node')

        # Get ros params
        self.gdino_checkpoint  = rospy.get_param('~gdino_checkpoint', "")
        self.gdino_config      = rospy.get_param('~gdino_config', "")
        self.sam_image_encoder = rospy.get_param('~sam_image_encoder', "")
        self.sam_mask_decoder  = rospy.get_param('~sam_mask_encoder', "")
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/aligned_depth_to_color/camera_info')
        self.color_topic       = rospy.get_param('~color_topic', '/camera/color/image_raw')
        self.depth_topic       = rospy.get_param('~aligned_depth_topic')

        # load models
        self.model, self.predictor = vlm_utils.load_models(
            gdino_checkpoint=self.gdino_checkpoint,
            gdino_config=self.gdino_config,
            sam_image_encoder=self.sam_image_encoder,
            sam_mask_decoder=self.sam_mask_decoder
        )

        # Set other parameters        
        self.depth_scale = 0.001 # 1 unit in depth image = 1 mm, according to this https://github.com/IntelRealSense/realsense-ros/issues/277#issuecomment-525676873
        self.bridge = CvBridge()
        self.latest_color_frame = None
        self.latest_depth_frame = None
        
        # Create synchronized subscriber for color and depth
        color_sub = message_filters.Subscriber(self.color_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=10,
            slop=0.05
        )
        sync.registerCallback(self.image_callback) # handle synchronized callback

        # Create ros service to detect obj when called
        self.object_detect_srv = rospy.Service(
            'detect_obj_pose',
            DetectObjectPose,
            self.detect_obj_pose # handle service call
        )
        # Create publisher to publish object pose
        self.pose_pub = rospy.Publisher('object_pose', Pose, queue_size=10)
        # Create publisher to publish (annotated) Image
        self.img_pub = rospy.Publisher('annotated_image', Image, queue_size=10)
        # Subscribe to CameraInfo to get camera intrinsic
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, queue_size=10, callback=self.get_camera_intrinsics)

        rospy.loginfo('==== object_detect_node initialized! ====')

    def get_camera_intrinsics(self, camera_info) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Read camera intrinsics from CameraInfo topic"""
        if self.cam_mtx is None or self.dist_coeffs is None:
            self.cam_mtx = np.array(camera_info.K).reshape(3, 3)
            self.dist_coeffs = np.array(camera_info.D)
        return self.cam_mtx, self.dist_coeffs

    def image_callback(self, color_msg, depth_msg):
        try:
            # Check for empty data
            if len(color_msg.data) == 0 or len(depth_msg.data) == 0:
                rospy.logwarn("Received empty image message(s), skipping.")
                return

            color_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough') # i don't know why but don't change this encoding 
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough') # usually 16UC1
            self.latest_color_frame = color_img
            self.latest_depth_frame = depth_img

        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def detect_obj_pose(self, req: DetectObjectPoseRequest):
        caption = req.caption
        rospy.loginfo(f'Received request with caption: {caption}')
        
        color_img = self.latest_color_frame.copy() # for gdino
        depth_img = self.latest_depth_frame.copy()
        pil_img = PIL.Image.fromarray(color_img.copy()) # for sam and visualization
        
        # Detect bbox with gdino
        bboxes, labels = self.model.predict_with_caption(
            image=color_img,
            caption=caption
        )

        # Detect mask from bounding box with sam
        sam_result = vlm_utils.predict_with_bbox(
            bboxes=bboxes, predictor=self.predictor, img=pil_img)
        masks = sv.Detections.from_sam(sam_result=sam_result)
        vlm_utils.annotate(pil_img, bboxes=bboxes, labels=labels, masks=masks)
        
        # Convert pil_img to ros_img
        cv_img = np.array(pil_img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        ros_img = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        # Publish annotated image
        self.img_pub.publish(ros_img)
        
        obj_list = []
        if len(bboxes) == 0:
            success = False
            message = f"Couldn't detect any object relating to this command: {caption}"
        else:
            success = True
            for idx, result in enumerate(sam_result):
                mask = result['segmentation'] # binary mask
                pcl = vlm_utils.get_pcl_from_mask(
                    mask, depth=depth_img, cam_mtx=self.cam_mtx, depth_scale=self.depth_scale)
                axes, centroid = vlm_utils.estimate_axes_from_pcl(pcl)
                T = vlm_utils.axes2matrix(axes=axes, centroid=centroid)
                pose = calib_utils.matrix2pose(T)
                obj = {
                    'label': labels[idx],
                    'pose': pose
                }
                obj_list.append(obj)
            message = f"Detected object: {labels}"
        
        rospy.loginfo(f'detected object: {obj_list}')
        return DetectObjectPoseResponse(
            poses = [obj['pose'] for obj in obj_list],
            labels = [obj['label'] for obj in obj_list],
            success=success,
            message=message,
        )

if __name__ == "__main__":
    node = ObjectDetectNode()
    rospy.spin()