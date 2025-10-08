import numpy as np
import yaml
import cv2

# ===============================
# Utility functions
# ===============================
def rotation_vector_to_matrix(rvec):
    """Convert OpenCV rotation vector to rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return R

def make_transform(R, t):
    """Construct 4x4 homogeneous transform from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def invert_transform(T):
    """Compute inverse of a homogeneous transform."""
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def save_transform_to_yaml(T, path):
    """Save transform matrix to a YAML file."""
    data = {
        'rotation_matrix': T[:3, :3].tolist(),
        'translation': T[:3, 3].tolist()
    }
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"[INFO] Saved transform to {path}")

def load_transform_from_yaml(path):
    """Load 4x4 transform matrix from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    R = np.array(data['rotation_matrix'])
    t = np.array(data['translation'])
    return make_transform(R, t)

# ===============================
# Main process
# ===============================
if __name__ == "__main__":
    rvec = np.array(
        [-1.53865472,  1.81244947, -0.25123752]
    )  
    tvec = np.array(
        [ 0.18347712, -0.13029944,  0.39182217]
    )

    # Build camera→aruco transform
    R_cam_aruco = rotation_vector_to_matrix(rvec)
    T_cam_aruco = make_transform(R_cam_aruco, tvec)

    # --- Load aruco→robot transform (pre-calculated) ---
    # YAML file format:
    # rotation_matrix:
    #   - [1, 0, 0]
    #   - [0, 1, 0]
    #   - [0, 0, 1]
    # translation: [0.1, 0.0, 0.2]
    T_aruco_robot = load_transform_from_yaml("aruco_to_robot.yaml")

    # --- Compute camera→robot transform ---
    T_cam_robot = T_cam_aruco @ T_aruco_robot

    # --- Save camera→robot transform ---
    save_transform_to_yaml(T_cam_robot, "camera_to_robot.yaml")

    print("\n[INFO] Camera → Robot transform:")
    print(T_cam_robot)

    # Optional: also compute robot→camera if needed
    T_robot_cam = invert_transform(T_cam_robot)
    save_transform_to_yaml(T_robot_cam, "robot_to_camera.yaml")

    print("\n[INFO] Robot → Camera transform:")
    print(T_robot_cam)
