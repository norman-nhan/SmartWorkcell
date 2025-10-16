import tf.transformations
from geometry_msgs.msg import Pose
import yaml
import numpy as np
from typing import Tuple, List
import cv2

def save_camera_calibration(path, cam_mtx, dist_coeffs=None, overall_rms=None, reproj_err_mean=None, reproj_err_total=None):
    data = {
        'camera_matrix': cam_mtx.tolist(),
        'dist_coeffs': dist_coeffs.flatten().tolist(),
        'overall_rms': overall_rms,
        'mean_reprojection_errors': reproj_err_mean,
        'total_reprojection_errors': reproj_err_total
    }
    with open(path, 'w') as f:
        yaml.dump(
            data,
            f,
            sort_keys=False, 
            default_flow_style=None,
            width=120,
            indent=2
        )
    print(f'[INFO] Saved camera intrinsic to {path}')

def load_camera_intrinsic(path) -> Tuple[np.ndarray, np.ndarray]:
    """Return camera matrix and distortion coefficients from a yaml file."""
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

def vectors2matrix(rvec, tvec):
    """Convert rotation vector and translation vector to 4x4 homogeneous transform."""
    R = rvec2matrix(rvec)
    T = make_transform_matrix(R, tvec)
    return T

def invert_transform(T):
    """Compute inverse of a homogeneous transform."""
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def save_multi_transforms(ids: List[int], T_list: List[np.ndarray], path):
    """Save multiple transform matrices to a YAML file."""
    data = {}
    for id, T in zip(ids, T_list):
        key = str(int(id))  # ensure YAML key is like '0', '5'
        data[key] = {'transform_mtx': T.tolist()}

    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=None, width=120, indent=2)
    print(f"[INFO] Saved transforms to {path}")

def load_multi_transforms(path):
    """Load transform matrices from a YAML file.
    
    Returns
    -------
    ids : list[int]
        List of marker IDs.
    T_list : list[np.ndarray]
        List of 4x4 transformation matrices.
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    ids, T_list = [], []
    for key, value in data.items():
        ids.append(int(key))
        T_list.append(np.array(value['transform_mtx'], dtype=float))
    return ids, T_list

def matrix2pose(T: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x = T[0, 3]
    pose.position.y = T[1, 3]
    pose.position.z = T[2, 3]
    
    quat = tf.transformations.quaternion_from_matrix(T)
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    
    return pose