import time
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse

class RealsenseCameraNode():
    def __init__(self, serial_number=None, image_save_dir="images/test", save_depth=False):
        self.ctx = rs.context()
        devices = self.ctx.query_devices()
        if len(devices) == 0:
            print("[WARN] No realsense devices found!")
        else:
            print(f"[INFO] {len(devices)} RealSense device(s) connected.\n")
            for i, dev in enumerate(devices):
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                firmware = dev.get_info(rs.camera_info.firmware_version)
                print(f"[{i}] {name}")
                print(f"    Serial Number: {serial}")
                print(f"    Firmware: {firmware}")

        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.colorizer = rs.colorizer() # Create colorizer for depth map
        if serial_number is not None:
            self.set_device(serial_number)
        
        # Enable color and depth streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # depth in 16bit format

        # Create folder to save images
        self.image_save_dir = image_save_dir
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

        self.save_depth = save_depth

    def set_device(self, serial) -> None:
        if isinstance(serial, int):
            serial = str(serial)
        
        self.config.enable_device(serial)
        print(f"[INFO] Device with seria number: {serial} was set.")
    
    def save_image(self, color_image, depth_image=None, depth_map=None):
        color_path = os.path.join(self.image_save_dir, f"color_{int(time.time() * 1000)}.png")
        cv2.imwrite(color_path, color_image)
        print(f"[INFO] Saved color image to {color_path}")

        if self.save_depth:
            depth_path = os.path.join(self.image_save_dir, f"depth_{int(time.time() * 1000)}.png")
            depth_map_path = os.path.join(self.image_save_dir, f"depth_{int(time.time() * 1000)}.npy")
            cv2.imwrite(depth_path, depth_image)
            np.save(depth_map_path, depth_map)
            print(f"[INFO] Saved colored depth image to {depth_path} and depth map to {depth_map_path}")

    def streaming(self, callback=None):
        # Start streaming
        self.pipeline.start(self.config)
        profile = self.pipeline.get_active_profile()
        device = profile.get_device()
        print(f"\n[INFO] Streaming device: {device.get_info(rs.camera_info.name)} (Serial: {device.get_info(rs.camera_info.serial_number)})")

        try:
            while profile is not None:
                # wait for a pair of depth and color
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
                depth_map = np.asanyarray(depth_frame.get_data())

                # Detect aruco pose callback
                if callback:
                    callback(color_image)

                # Stack images side by side
                images = np.hstack((color_image, depth_image))
                # Show images
                cv2.imshow('RealSense RGB + Depth', images)

                key = cv2.waitKey(1) & 0xFF # Get key every 1ms
                # Press 'q' to exit
                if key == ord('q'):
                    break
                # Press 's' to save images
                elif key == ord('s'):
                    self.save_image(color_image, depth_image=depth_image, depth_map=depth_map)
                elif key == ord('p'):
                    self.pipeline.stop()
        except KeyboardInterrupt:
            pass
        finally:
            # Stop streaming
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("[INFO] All processes shut down successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial_number", type=str)
    parser.add_argument("-i", "--image_save_dir", type=str, default="images/test")
    parser.add_argument("--save_depth", action="store_true")
    args = parser.parse_args()
    
    rsCam = RealsenseCameraNode(
        serial_number=args.serial_number, image_save_dir=args.image_save_dir, save_depth=args.save_depth
    )
    rsCam.streaming()