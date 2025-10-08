#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
from moveit_msgs.msg import DisplayRobotState, RobotState

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("show_goal_pose", anonymous=True)

    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander("manipulator")

    # Define a goal pose
    pose_goal = Pose()
    pose_goal.position.x = 0.4
    pose_goal.position.y = 0.0
    pose_goal.position.z = 0.3
    pose_goal.orientation.w = 1.0

    # Apply target
    group.set_pose_target(pose_goal)
    
    # Convert that target into a RobotState
    target_state = group.get_joint_value_target()
    robot_state = RobotState()
    robot_state.joint_state.name = group.get_active_joints()
    robot_state.joint_state.position = target_state

    # Publisher for RViz goal display
    pub = rospy.Publisher("/move_group/display_robot_state", DisplayRobotState, queue_size=10)

    msg = DisplayRobotState()
    msg.state = robot_state

    rospy.sleep(1.0)  # wait for RViz subscriber
    rospy.loginfo("Publishing goal state to RViz...")

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        group.plan()
        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    main()
