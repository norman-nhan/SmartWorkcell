#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import yaml
import moveit_msgs.msg
import copy


        
def main():
    moveit_commander.roscpp_initialize(sys.argv) # take args from cli
    rospy.init_node("move2goal", anonymous=True)

    # Initialize robot and move_group
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    move_group = moveit_commander.MoveGroupCommander("manipulator")  # move_group name in SRDF
    
    planning_frame = move_group.get_planning_frame()
    print("============ Planning frame: %s" % planning_frame)

    # We can also print the name of the end-effector link for this group:
    eef_link = move_group.get_end_effector_link()
    print("============ End effector link: %s" % eef_link)

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print("============ Available Planning Groups:", robot.get_group_names())

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print("============ Printing robot state")
    print(robot.get_current_state())
    print("")

    # Create target pose
    # Pose must be reachable and collision-free
    # If it is at a singularity, MoveIt will fail to plan
    # Read test_pose.yml for a sample pose
    with open("test_pose.yml", "r") as f:
        config = yaml.safe_load(f)
        print(f"config: {config}")
    goal_pose_cfg = config["pose2"]
    goal_pose = geometry_msgs.msg.Pose()
    goal_pose.position.x = goal_pose_cfg["position"]["x"]
    goal_pose.position.y = goal_pose_cfg["position"]["y"]
    goal_pose.position.z = goal_pose_cfg["position"]["z"]
    goal_pose.orientation.x = goal_pose_cfg["orientation"]["x"]
    goal_pose.orientation.y = goal_pose_cfg["orientation"]["y"]
    goal_pose.orientation.z = goal_pose_cfg["orientation"]["z"]
    goal_pose.orientation.w = goal_pose_cfg["orientation"]["w"]

    # Send goal
    move_group.set_pose_target(goal_pose)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rospy.loginfo("Planning...")
        success, _, _, _ = move_group.plan()
        if success:
            rospy.loginfo("Planning succeed.")
            # break
        else:
            rospy.logwarn("Planning failed. Retry...")
        rate.sleep()
    
    # rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.signal_shutdown("shutdown")