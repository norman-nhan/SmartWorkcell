import pyrealsense2 as rs
from groundingdino.util.inference import Model
from nanosam.utils.predictor import Predictor
import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d
import supervision as sv
import yaml
import matplotlib.pyplot as plt

def load_models(gdino_checkpoint='config/gdino/weights/groundingdino_swinb_cogcoor.pth',
                gdino_config='config/gdino/GroundingDINO_SwinB_cfg.py',
                sam_image_encoder='config/nanosam/data/resnet18_image_encoder.engine',
                sam_mask_decoder='config/nanosam/data/mobile_sam_mask_decoder.engine'
                ):
    """Load groundingdino and nanosam
    
    Returns
    -------
    - gdino_model: groundingdino model
    - sam_predictor: nanosam predictor"""
    # load gdino
    gdino_model = Model(
        model_checkpoint_path=gdino_checkpoint,
        model_config_path=gdino_config
    )
    # load nanosam predictor
    sam_predictor = Predictor(
        image_encoder_engine=sam_image_encoder,
        mask_decoder_engine=sam_mask_decoder
    )
    return gdino_model, sam_predictor

def bbox2points(bbox):
    """Retrive bounding box top-left and bottom-right
    
    Returns
    -------
    - points: np.ndarray
    - points_labels: np.ndarray
        - 0: background point,
        - 1: foreground point,
        - 2: bounding box top-left
        - 3: bounding box bottom-right
    """
    points = np.array([
        [bbox[0], bbox[1]], # top-left
        [bbox[2], bbox[3]]  # bottom-right
    ])
    # point_labels is used to define what kind of points is returned
    point_labels = np.array([2, 3]) # top-left and bottom-right

    return points, point_labels

def get_pcl_from_mask(binary_mask, depth_map, intrinsics):
    """Get 3D point cloud within a given mask.
    
    Args
    ----
    binary_mask: np.ndarray
        detected object mask, shape (H, W), dtype=bool
    depth_map: np.ndarray
        contains depth values of an image, shape (H, W), dtype=float32, unit in meters
    intrinsics: dict
        camera intrinsics, contains fx, fy, cx, cy, depth_scale.
    
    Returns
    -------
    pcl: np.ndarray
        point cloud within the mask, shape (N, 3), dtype=float32, unit in meters. 
    """
    ys, xs = np.where(binary_mask) # Get masked pixel coordinates
    depth = depth_map[ys, xs] * intrinsics["depth_scale"]# Get depth

    pcl = np.zeros((len(xs), 3), dtype=np.float32) # (N, 3) = (len(xs), 3)
    pcl[:, 0] = (xs - intrinsics["cx"]) * depth / intrinsics["fx"]
    pcl[:, 1] = (ys - intrinsics["cy"]) * depth / intrinsics["fy"]
    pcl[:, 2] = depth
    return pcl

def get_binary_mask_using_nanosam(bbox, predictor):
    """Retrive binary mask from a bounding box using nanosam.
    
    Returns
    -------
    - binary_mask: True/False mask.
    
    Args
    ----
    - bbox: bounding box
    - sam_predictor: nanosam predictor"""
    points, point_labels = bbox2points(bbox.xyxy[0])
    mask, _, _ = predictor.predict(points, point_labels)
    binary_mask = (mask[0, 0] > 0).detach().cpu().numpy() # True/False mask, shape: (H, W)
    return binary_mask

def get_annotate_image(image, bboxes, labels, sv_masks):
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    out_image = box_annotator.annotate(scene=image.copy(), detections=bboxes)
    out_image = label_annotator.annotate(scene=out_image, detections=bboxes, labels=labels)
    out_image = mask_annotator.annotate(scene=out_image, detections=sv_masks)
    plt.imshow(out_image)
    
    return out_image

def enforce_right_hand_rule(axes):
    """Enfore a given set of axes to be right-handed.
    
    Args
    ----
    axes: np.ndarray 
        An (3, 3) array, contains 3 principal axes as rows.
        
        Example: axes: (3, 3), [[-0.08448856  0.09596848  0.9917922 ]
                [ 0.994619    0.06801371  0.07814818]
                [-0.0599557   0.993058   -0.10119846]]
    
    Returns
    -------
    axes: np.ndarray
        A set of axes following right-hand rule.
    """
    x_axis = axes[0]
    y_axis = axes[1]
    z_axis = axes[2]
    z_cross = np.cross(x_axis, y_axis)
    if np.dot(z_cross, z_axis) < 0:
        # Flip the third axis to ensure right-handedness
        axes[2] = -axes[2]
    return axes

def get_pca_components_of_pcl(pcl):
    """Return right-handed PCA components of given point cloud."""
    pca = PCA(n_components=3)
    pca.fit(pcl)
    axes = pca.components_
    axes = enforce_right_hand_rule(axes) # Ensure result is right-handed

    return axes

def remove_outliers_from_pcl(pcl, nb_neighbors=20, std_ratio=2.0):
    """Return outliers/noise removed pcl."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcl_denoised = np.asarray(cl.points)

    return pcl_denoised

def draw_pca_axes_on_pcl(pcl, ax_length=0.05):
    """Visualize enforced right-handed PCA components on a set of pcl."""
    # 1. Clean pcl
    denoised_pcl = remove_outliers_from_pcl(pcl)
    
    # 2. Apply PCA
    axes = get_pca_components_of_pcl(denoised_pcl)
    centroid = np.mean(denoised_pcl, axis=0) # center point of pcl

    # 3. Visualize with Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(denoised_pcl)

    # Create lines for axes
    lines = []
    colors = [[1,0,0], [0,1,0], [0,0,1]]  # X=red, Y=green, Z=blue
    points = [centroid]
    for i in range(3):
        axis_end = centroid + axes[i] * ax_length
        points.append(axis_end)
        lines.append([0, i+1])  # from centroid to axis end

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd, line_set])