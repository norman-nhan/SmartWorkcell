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

def save_translation_matrix2yaml(T, path):
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
    """rvec and tvec are VECTORS!
    I'm confusing whether rvec consists [yaw pitch roll] or not. 
    Apparently, rvec represents how much the tip of each marker's axis differs from the origin vector.
    Let's take vectorAB + vectorBC = vectorAC as example. Where B is the tip of origin vector and C is the tip of new vector.
    In order to form vectorAC which points to new position, we need to add vectorBC with origin vectorAB.
    The magnitude of vectorBC becomes the angle of vectorAC and vectorAB
    NOTE: KEEP IN MIND THAT camera is the origin coordinate system. Meaning that vectorAC is one of camera coordinate axes."""
    # Marker pose in camera frame
    tmp_rotation_matrix_cam_marker=np.array([[0, 0, -1],
                                            [0, -1, 0],
                                            [-1, 0, 0]], dtype=np.float64)
    rvec_cam_marker, _ = cv2.Rodrigues(tmp_rotation_matrix_cam_marker)
    # rvec_cam_marker = np.array( # vector/matrix_A_B, etc. meaning A coordinates to B coordinates vector/matrix
    #     # [-1.53865472,  1.81244947, -0.25123752]
    # )  
    tvec_cam_marker = np.array(
        # [ 0.18347712, -0.13029944,  0.39182217]
        [0.1, 0.0, 0.0] # in front of camera 10cm
    )

    # Convert vector to translation matrix for 3d rigid transformation
    R_cam_marker = rotation_vector_to_matrix(rvec_cam_marker)
    T_cam_marker = make_transform(R_cam_marker, tvec_cam_marker)

    # --- Load marker-robot transform (pre-calculated) ---
    # Example: YAML file format:
    # rotation_matrix:
    #   - [1, 0, 0]
    #   - [0, 1, 0]
    #   - [0, 0, 1]
    # translation: [0.1, 0.0, 0.2]
    T_marker_robot = load_transform_from_yaml("../config/marker_robot_translation_matrix.yaml")

    # --- Compute camera→robot transform ---
    T_cam_robot = T_cam_marker @ T_marker_robot

    # --- Save camera→robot transform ---
    save_translation_matrix2yaml(T_cam_robot, "cam_robot_translation_matrix.yaml")

    print("\n[INFO] Camera → Robot transform:")
    print(T_cam_robot)

    # Optional: also compute robot→camera if needed
    T_robot_cam = invert_transform(T_cam_robot)
    save_translation_matrix2yaml(T_robot_cam, "robot_cam_translation_matrix.yaml")

    print("\n[INFO] Robot → Camera transform:")
    print(T_robot_cam)
