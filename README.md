# SmartWorkcell
## Installation
### This package
```bash
cd $YOUR_WS
git clone https://github.com/norman-nhan/SmartWorkcell.git
cd SmartWorkcell
pip install -r requirements.txt
python3 setup.py sdist
pip install -e .
```
Replace `$YOUR_WS` which your working directory.
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
