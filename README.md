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
### (Optional) Camera calibration
For this project I'm using RealSense camera factory calibrated data.

**TODO**: 
Make a pipeline that open camera 
-> detect marker (every 10mins or after service call)
-> compute T_cam_marker 
-> read T_marker_robot 
-> compute T_robot_cam
-> detect object 
-> compute T_cam_object 
-> use T_robot_cam computed before to compute T_robot_object
-> Convert T_robot_object to Pose() that MoveIt can use
-> Define Pick & Place with MoveIt (Next step!)
