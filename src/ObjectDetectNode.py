#!/usr/bin/env python3

import PIL
import cv2
import numpy as np
import supervision as sv
from typing import Tuple, List
import message_filters
from pysmartworkcell import (
    vlm_utils, 
    calibration_utils as calib_utils
)
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from smartworkcell.srv import DetectObjectPose, DetectObjectPoseResponse, DetectObjectPoseRequest

class ObjectDetectNode():
    def __init__(self):
        rospy.init_node('object_detect_node')
        # Create ros service
        self.object_detect_srv = rospy.Service(
            'object_detect_service',
            DetectObjectPose,
            self.detect_obj_pose
        )
        # Create publisher pubs Pose
        self.pose_pub = rospy.Publisher('object_pose', Pose, queue_size=10)
        # Create publisher pubs (annotated) Image
        self.img_pub = rospy.Publisher('annotated_image', Image, queue_size=10)
        
        # Load rosparams
        self.gdino_checkpoint = rospy.get_param('gdino_checkpoint', "")
        self.gdino_config = rospy.get_param('gdino_config', "")
        self.sam_image_encoder = rospy.get_param('sam_image_encoder', "")
        self.sam_mask_decoder = rospy.get_param('sam_mask_encoder', "")

        self.camera_config = rospy.get_param('camera_config')
        self.color_topic = rospy.get_param('color_topic')
        self.depth_topic = rospy.get_param('aligned_depth_topic')
        # Create synchronized subscriber for color and depth
        color_sub = message_filters.Subscriber(self.color_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=10,
            slop=0.05
        )
        sync.registerCallback(self.image_callback)
        # converting ROS and OpenCV
        self.bridge = CvBridge()
        self.latest_color_frame = None
        self.latest_depth_frame = None
        
        # load models
        self.model, self.predictor = vlm_utils.load_models(
            gdino_checkpoint=self.gdino_checkpoint,
            gdino_config=self.gdino_config,
            sam_image_encoder=self.sam_image_encoder,
            sam_mask_decoder=self.sam_mask_decoder
        )
        # load camera intrinsic
        self.cam_mtx, self.dist_coeffs, self.depth_scale = calib_utils.load_camera_intrinsic(self.camera_config)
        
        self.wait_for_images()
        
        rospy.loginfo('==== object_detect_node initialized! ====')
    
    def image_callback(self, color_msg, depth_msg):
        try:
            # rospy.loginfo(f"Color msg encoding: {color_msg.encoding}, width: {color_msg.width}, height: {color_msg.height}")
            # rospy.loginfo(f"Depth msg encoding: {depth_msg.encoding}, width: {depth_msg.width}, height: {depth_msg.height}")

            # Check for empty data
            if len(color_msg.data) == 0 or len(depth_msg.data) == 0:
                rospy.logwarn("Received empty image message(s), skipping.")
                return

            color_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough') # i don't know why but don't change this encoding 
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')  # usually 16UC1
            self.latest_color_frame = color_img
            self.latest_depth_frame = depth_img

        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")

    def wait_for_images(self, timeout=10.0):
        rospy.loginfo("Waiting for synchronized color and depth images...")
        start_time = rospy.Time.now().to_sec()
        while (self.latest_color_frame is None or self.latest_depth_frame is None) and \
            (rospy.Time.now().to_sec() - start_time < timeout):
            rospy.sleep(0.1)
        if self.latest_color_frame is None or self.latest_depth_frame is None:
            raise RuntimeError("Timeout: No color/depth frames received.")
        rospy.loginfo("Images received!")

    
    def detect_obj_pose(self, req: DetectObjectPoseRequest):
        caption = req.caption
        rospy.loginfo(f'Received request with caption: {caption}')
        
        color_img = self.latest_color_frame
        depth_img = self.latest_depth_frame
        pil_img = PIL.Image.fromarray(color_img)
        bboxes, labels = self.model.predict_with_caption(
            image=self.latest_color_frame,
            caption=caption
        )

        # Detect mask with bounding box
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
                mask = result['segmentation']
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