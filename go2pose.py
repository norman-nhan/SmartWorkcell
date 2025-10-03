#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import moveit_msgs.msg

class Move2Goal:
    def __init__(self):
        # Initialize moveit_commander and rospy node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_to_pose_example", anonymous=True)
        
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("manipulator") # descripted in robot srdf file
        self.display_traj_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                 moveit_msgs.msg.DisplayTrajectory,
                                                 queue_size=20)

        # ======================= #
        # PRINT BASIC INFORMATION #
        # ======================= #
        self.planning_frame = self.move_group.get_planning_frame()
        print(f"========= Reference frame: {self.planning_frame}")
        self.eef_link = self.move_group.get_end_effector_link()
        print(f"========= End effector: {self.eef_link}")
        self.group_names = self.robot.get_group_names()
        print(f"========= Robot Groups: {self.group_names}")
        print(f"========= Printing robot state")
        print(self.robot.get_current_state())
        print("")

        # Misc variables
        self.box_name = ''

    def publish_traj(self, traj):
        display_traj = moveit_msgs.msg.DisplayTrajectory()
        display_traj.trajectory_start = self.robot.get_current_state()
        display_traj.trajectory.append(traj)
        self.display_traj_publisher.publish(display_traj)
    
    def go2pose(self, goal_pose):
        self.move_group.set_pose_target(goal_pose)
        rospy.loginfo(f"Set goal pose: {goal_pose}")
        rospy.loginfo(f"Planning...")
        plan = self.move_group.plan()

        if plan[0]:
            self.publish_traj(plan[1])
            
            input("Press `Enter` to execute plan")
            self.move_group.execute(plan[1], wait=True)
        else:
            rospy.loginfo("Planning failed.")

def main():
    try:  
        input("========= Press `Enter` to start demo")
        move2goal = Move2Goal()
        # Create target pose
        # Pose must be reachable and collision-free
        # If it is at a singularity, MoveIt will fail to plan
        goal = geometry_msgs.msg.Pose()
        goal.position.x = 0.4   # meters in robot base frame
        goal.position.y = 0.0
        goal.position.z = 0.3
        goal.orientation.x = 0.0
        goal.orientation.y = 1.0
        goal.orientation.z = 0.0
        goal.orientation.w = 0.0
        input("========= Press `Enter` to execute a movement using goal")
        move2goal.go2pose(goal_pose=goal)
        print("========= Demo completed!")
    
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return
    finally:
        rospy.loginfo("Shutting down.")

if __name__ == "__main__":
    main()