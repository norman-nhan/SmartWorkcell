#!/usr/bin/env python3
import rospy
import sys
import moveit_commander
import actionlib
from smartworkcell.msg import (
    MoveJoint2ObjPoseAction,
    MoveJoint2ObjPoseFeedback,
    MoveJoint2ObjPoseResult
)
from geometry_msgs.msg import Pose

class MoveJoint2ObjPoseServer():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_arm_node')
        
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander('manipulator')

        # Print basic info
        self.planning_frame = self.move_group.get_planning_frame()
        self.eff_link = self.move_group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        rospy.loginfo(f'Reference frame: {self.planning_frame}')
        rospy.loginfo(f'End effector: {self.eff_link}')
        rospy.loginfo(f'Robot Groups: {self.group_names}')
        rospy.loginfo(f'Robot State: \n{self.robot.get_current_state()}')
        rospy.loginfo('move_arm_node initialization completed!')

        # Create publisher, subscriber, action
        self.server = actionlib.SimpleActionServer(
            'move_joint_2_obj_pose',
            MoveJoint2ObjPoseAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self.server.start()
        rospy.loginfo('MoveJoint2ObjPose action server ready!')
    
    def execute_cb(self, goal):
        feedback = MoveJoint2ObjPoseFeedback()
        result = MoveJoint2ObjPoseResult()
        
        rospy.loginfo(f'Received target pose:\n{goal.target_pose}')
        
        feedback.state = 'Planning to target'
        self.server.publish_feedback(feedback)
        self.move_group.set_pose_target(goal.target_pose)
        plan = self.move_group.plan()
        if not plan[0]:
            result.success = False
            result.message = 'Planning failed'
            self.server.set_aborted(result)
            
        feedback.state = 'Executing'
        self.server.publish_feedback(feedback)
        self.move_group.execute(plan[1], wait = True)
        
        result.success = True
        result.message = 'Task completed!'
        self.server.set_succeeded(result)
        rospy.loginfo('Task completed!')
        
    def move_joint_to_home_pos(self):
        self.move_group.set_named_target('home')
        plan = self.move_group.go(wait=True)
        self.move_group.stop() # stop after executation is finished
        rospy.loginfo("Move to home position:", plan)
        
        
if __name__ == '__main__':
    MoveJoint2ObjPoseServer()
    rospy.spin()
    # try:
    #     node = MoveJoint2ObjPoseServer()
    #     node.move_joint_to_home_pos()
    # except:
    #     pass
    # finally:
    #     moveit_commander.roscpp_shutdown()