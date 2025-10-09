import yaml
import numpy as np
from typing import Tuple
import cv2

def get_camera_intrinsic(path) -> Tuple[np.ndarray, np.ndarray]:
    """Return  2 np.ndarray from a yaml file:
    - camera matrix and 
    - distortion coefficients 
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return (np.array(data["camera_matrix"]), np.array(data["dist_coeffs"]))

def rvec2matrix(rvec):
    """Convert OpenCV rotation vector to rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return R

def make_transform_matrix(R, t):
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

def save_transform_matrix(T, path):
    """Save transform matrix to a YAML file."""
    data = {
        'rotation_matrix': T[:3, :3].tolist(),
        'translation': T[:3, 3].tolist()
    }
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=None, width=120, indent=2)
    print(f"[INFO] Saved transform to {path}")

def load_transform_from_yaml(path):
    """Load 4x4 transform matrix from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    R = np.array(data['rotation_matrix'])
    t = np.array(data['translation'])
    return make_transform_matrix(R, t)