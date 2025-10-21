# SmartWorkcell
This package using GroundingDINO and NanoSAM to detect object pose and MoveIt to control robot arm (ur3e). It includes both python package and ROS package.
## Dependencies
1. [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO.git)
2. [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam.git)
3. [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
4. [IntelRealSense](https://github.com/IntelRealSense/realsense-ros)
## Build
### For `pysmartworkcell` python package 
```bash
cd $YOUR_WS
git clone https://github.com/norman-nhan/SmartWorkcell.git
cd smartworkcell
pip install -r requirements.txt
python3 setup.py sdist
pip install -e .
```
Replace `$YOUR_WS` which your working directory.
### For `smartworkcell` ros package
```bash
cd $YOUR_WS
catkin_make
# pyenv activate SmartWorkcell # (optional) this is my virtualenv 
source devel/setup.bash
```
> [!TIP]
> When working with virtualenv, make sure your virtualenv can see your local environment which has ROS installed.
> You can check it by using `echo $PYTHONPATH`. If `$PYTHONPATH` shows ros's dist-packages then you're good to go!
## Getting start
- Run realsense camera node
  ```bash
  roslaunch smartworkcell rs_d435_node.launch
  ```
- Run object detection node (groundingdino and nanosam)
  ```bash
  roslaunch smartworkcell object_detect_node.launch
  ```
- Run camera-robot calibration node
  ```bash
  roslaunch camera_calibration_node.launch
  ```
- Run all nodes
  ```bash
  roslaunch smartworkcell smartworkcell_test.launch
  ```

## Development
> [!TIP]
> **Autocomplete for ROS-python packages**
>
> Add `"/opt/ros/noetic/lib/python3/dist-packages"` to `python.analysis.extraPaths` in `settings.json`.
> Example:
> ```json
> "python.analysis.extraPaths": ["/opt/ros/noetic/lib/python3/dist-packages"]
> ```
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
## Debug
### ARUCO MARKER DETECTION
#### For RealSense camera
Please don't use the calibration result from `calibration.py`. It's not correct. It is the reason causing flicking axes and glitches.
Highly recommend using the camera intrinsic using `pyrealsense` if you're using RealSense camera.
#### For general camera
##### 2025/10/8 Updates
I have done several tests with different chessboard size and I found that for mac webcam you can use small chessboard size (Checkerboard-A4-40mm-6x4) to increase calibration result.