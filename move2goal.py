#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import yaml

def main():
    moveit_commander.roscpp_initialize(sys.argv) # take args from cli
    rospy.init_node("move2goal", anonymous=True)

    # Initialize robot and group
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("manipulator")  # group name in SRDF

    # Create target pose
    # Pose must be reachable and collision-free
    # If it is at a singularity, MoveIt will fail to plan
    # Read test_pose.yml for a sample pose
    with open("test_pose.yml", "r") as f:
        config = yaml.safe_load(f)
        print(f"config: {config}")
    goal_pose_cfg = config["pose"]
    goal_pose = geometry_msgs.msg.Pose()
    goal_pose.header.frame_id = goal_pose_cfg["header"]["frame_id"]
    goal_pose.covariance = goal_pose_cfg["covariance"]
    goal_pose.position.x = goal_pose_cfg["position"]["x"]
    goal_pose.position.y = goal_pose_cfg["position"]["y"]
    goal_pose.position.z = goal_pose_cfg["position"]["z"]
    goal_pose.orientation.x = goal_pose_cfg["orientation"]["x"]
    goal_pose.orientation.y = goal_pose_cfg["orientation"]["y"]
    goal_pose.orientation.z = goal_pose_cfg["orientation"]["z"]
    goal_pose.orientation.w = goal_pose_cfg["orientation"]["w"]

    # Send goal
    group.set_pose_target(goal_pose)

    # plan = group.go(wait=True)
    plan = group.plan()

    if plan and plan[0]:
        rospy.loginfo("Plan found! Visualizing in RViz...")
    else:
        rospy.logwarn("Planning failed!")

    group.stop()
    group.clear_pose_targets()

    rospy.loginfo("Motion complete.")

if __name__ == "__main__":
    main()
