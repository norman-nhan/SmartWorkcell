#!/usr/bin/env python3
import rospy
import actionlib
from geometry_msgs.msg import Pose
from smartworkcell.msg import MoveJoint2ObjPoseAction, MoveJoint2ObjPoseGoal
from smartworkcell.srv import DetectObjectPose, DetectObjectPoseRequest
class MoveArmNode():
    def __init__(self):
        rospy.init_node('move_arm_node')
        
        # Get ros parameters
        self.obj_detect_service_name = rospy.get_param('obj_detect_service_name', default='/object_detect_service')
        # Wait for object detect service
        rospy.loginfo(f'Waiting for {self.obj_detect_service_name}...')
        rospy.wait_for_service(self.obj_detect_service_name)
        rospy.loginfo(f'{self.obj_detect_service_name} is available!')
        
        # Create a proxy to call the service
        self.detect_service = rospy.ServiceProxy(self.obj_detect_service_name, DetectObjectPose)
        
        # Wait for the moveit action server        
        rospy.loginfo('Waiting for move_joint_2_obj_pose_server...')
        self.client = actionlib.SimpleActionClient('move_joint_2_obj_pose', MoveJoint2ObjPoseAction)
        self.client.wait_for_server()
        rospy.loginfo("Action server connected!")
        
    def detect_obj(self, obj_name: str):
        try:
            req =DetectObjectPoseRequest()
            req.caption = obj_name
            res = self.detect_service(req)
            rospy.loginfo(f'Detected labels: {res.labels}')
            rospy.loginfo(f'Detected poses: {res.poses}')
            
            return res.poses
        except rospy.ServiceException as e:
            rospy.logerr(f'Service call failed: {e}')
            return []
        
    def send_goal(self, pose):
        goal = MoveJoint2ObjPoseGoal(target_pose=pose)
        self.client.send_goal(goal, feedback_cb=self.feedback_cb)
        self.client.wait_for_result()
        result = self.client.get_result()
        rospy.loginfo(f"Result: success={result.success}, message={result.message}")

    @staticmethod
    def feedback_cb(feedback):
        rospy.loginfo(f"Feedback: {feedback.state}")

if __name__ == "__main__":
    node = MoveArmNode()
    
    # --- Detect and move to first object ---
    poses = node.detect_obj("keyboard")
    if poses:
        node.send_goal(poses[0])
    else:
        rospy.logwarn("No poses received from detection service.")