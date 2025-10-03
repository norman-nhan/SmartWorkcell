# SmartWorkcell
This repo contains scripts (python files) for SmartWorkcell.
## Simulation
To run test and debug for ur3e in simulation (avoid real-world collision/remote-work) run these CLIs in order:
1. Open ur3e in Gazebo:
   ```bash
   roslaunch ur_gazebo ur3e_bringup.launch
   ```
2. Run MoveIt for motion planning:
   ```
   roslaunch ur3e_moveit_config moveit_planning_execution.launch sim:=True 
   ```
4. Open RViz:
   ```bash
   roslaunch ur3e_moveit_config moveit_rviz.launch 
   ```
