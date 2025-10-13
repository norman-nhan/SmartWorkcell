import pyrealsense2 as rs


def initialize_realsense_camera(serial:str =None):
    """Intialize realsense camera.
    
    Returns
    -------
    - pipeline
    - config:
    """
    # Get basic information
    ctx = rs.context()
    print_connected_devices_info(ctx)

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    if serial is not None:
        config.enable_device(serial)
        print(f"[INFO] Device with seria number: {serial} was set.")
    
    # Enable color and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    return pipeline, config

def print_connected_devices_info(ctx:rs.context):
    devices = ctx.query_devices()
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

def print_available_sensors(device:rs.device):
    print(f"[INFO] Device: {device.get_info(rs.camera_info.name)}")
    print("[INFO] Available sensors and streams:")

    for sensor in device.query_sensors():
        print(f"  - Sensor: {sensor.get_info(rs.camera_info.name)}")
        for stream_profile in sensor.get_stream_profiles():
            stream_type = stream_profile.stream_type()
            fmt = stream_profile.format()
            fps = stream_profile.fps()
            print(f"     • Stream: {stream_type}, Format: {fmt}, FPS: {fps}")