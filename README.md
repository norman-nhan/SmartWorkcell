# SmartWorkcell
## (Optional) Auto-completetion for ROS python libraries
1. Press `Ctrl + Shift + P` -> Choose `Preferences: Open Workspace Settings (JSON)`
2. Add `"/opt/ros/noetic/lib/python3/dist-packages"` to `python.analysis.extraPaths` in `settings.json`.
  
Example:
```json
{
    "python.analysis.extraPaths": [
        "./GroundingDINO",
        "./segment-anything",
        "/opt/ros/noetic/lib/python3/dist-packages"
    ],
    "python-envs.defaultEnvManager": "ms-python.python:system",
    "python-envs.pythonProjects": []
}
```
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
