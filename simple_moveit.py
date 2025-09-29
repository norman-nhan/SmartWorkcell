#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_to_pose_example", anonymous=True)

    # Initialize robot and group
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("manipulator")  # group name in SRDF

    # Create target pose
    # Pose must be reachable and collision-free
    # If it is at a singularity, MoveIt will fail to plan
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = 0.4   # meters in robot base frame
    pose_goal.position.y = 0.0
    pose_goal.position.z = 0.3
    pose_goal.orientation.x = 0.0
    pose_goal.orientation.y = 1.0
    pose_goal.orientation.z = 0.0
    pose_goal.orientation.w = 0.0

    # Send goal
    group.set_pose_target(pose_goal)

    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    rospy.loginfo("Motion complete.")

if __name__ == "__main__":
    main()
