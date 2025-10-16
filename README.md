# SmartWorkcell
## Installation
### This package
```bash
cd ~/nhan_ws/src/SmartWorkcell
python3 setup.py sdist
pip install -e dist/SmartWorkcell-0.0.1.tar.gz
```
### Realsense2
After installed realsense sdk from their website don't forget to add udev rule if it's not installed yet.
Download udev rule from this [link](https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules) then copy it to `/etc/udev/rules.d/` OR using this command:
```bash
cd ~/Downloads; wget https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules
sudo cp ~/Downloads/99-realsense-libusb.rules /etc/udev/rules.d/
```
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

I have tested with realsense camera and the result is also stable with calibration file from `calibration.py` with Checkerboard-A4-40mm-6x4
## Usage
