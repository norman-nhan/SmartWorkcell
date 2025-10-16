from scipy.spatial.transform import Rotation as scipyRot
from typing import List, Tuple
import torch
import pyrealsense2 as rs
from groundingdino.util.inference import Model
from nanosam.utils.predictor import Predictor
import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d
import supervision as sv
import matplotlib.pyplot as plt
from PIL import Image
from SmartWorkcell.calibration_utils import make_transform_matrix
from pathlib import Path

def get_project_info() -> Tuple[Path]:
    """Call this func from anywher to get the project root, config, io and src"""
    current_path = Path.cwd()

    # Search upward for the folder named "SmartWorkcell"
    for parent in current_path.parents:
        if (parent / "config").exists() and (parent / "src/SmartWorkcell").exists():
            PROJECT_ROOT = parent
            break
    else:
        raise RuntimeError("SmartWorkcell root not found")
    
    DEVICE = 'cuda'
    CONFIG_DIR = PROJECT_ROOT / "config"
    IO_DIR = PROJECT_ROOT / "io"
    SRC_DIR = PROJECT_ROOT / "src"
    print(f'project_root: {PROJECT_ROOT}\nconfig dir: {CONFIG_DIR}\nio dir: {IO_DIR}\nsrc dir: {SRC_DIR}')
    return PROJECT_ROOT, CONFIG_DIR, IO_DIR, SRC_DIR

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


def get_pcl_from_mask(binary_mask, depth, cam_mtx, device='cuda', depth_scale: float=1.0) -> torch.Tensor:
    """Get 3D point cloud within binary mask from depth directly on GPU.
    
    Args
    ----
    binary_mask: np.ndarray
        detected object mask, shape (H, W), dtype=bool
    depth: np.ndarray
        contains depth values of an image, shape (H, W), dtype=float32, unit in meters
    cam_mtx: np.ndarray or tensor
        3x3 amera matrix
    
    Returns
    -------
    pcl: tensor
        point cloud within the mask, shape (N, 3), dtype=float32, unit in meters. 
    """
    # Simplify camera matrix
    fx = cam_mtx[0,0]
    fy = cam_mtx[1,1]
    cx = cam_mtx[0,2]
    cy = cam_mtx[1,2]
    # Convert numpy to tensor
    binary_mask = torch.as_tensor(np.array(binary_mask, copy=True), device=device)
    depth = torch.as_tensor(np.array(depth, copy=True), device=device, dtype=torch.float32)

    # Get coordinates where mask is True
    ys, xs = torch.nonzero(binary_mask, as_tuple=True)
    # Get depth at those values
    depth = depth[ys, xs] * depth_scale

    # Compute point cloud (N, 3)
    pcl = torch.zeros((xs.shape[0], 3), device=device, dtype=torch.float32)
    pcl[:, 0] = (xs - cx) * depth / fx
    pcl[:, 1] = (ys - cy) * depth / fy
    pcl[:, 2] = depth
    
    return pcl # stay on gpu

def get_pcl_from_mask(binary_mask: np.ndarray, depth: np.ndarray, 
                      cam_mtx: np.ndarray, depth_scale: float=1.0) -> np.ndarray:
    """Get 3D point cloud within binary mask from depth directly on GPU.
    
    Args
    ----
    binary_mask: np.ndarray
        True/False mask
    depth: np.ndarray
        contains depth values of an image
    cam_mtx: np.ndarray
        3x3 camera matrix
    
    Returns
    -------
    pcl: np.ndarray
        point cloud within given mask
    """
    # Simplify camera matrix
    fx = cam_mtx[0,0]
    fy = cam_mtx[1,1]
    cx = cam_mtx[0,2]
    cy = cam_mtx[1,2]
    
    # Get coordinates where mask is True
    ys, xs = np.nonzero(binary_mask)
    # Get depth at those values
    depth = depth[ys, xs] * depth_scale

    # Compute point cloud (N, 3)
    pcl = np.zeros((xs.shape[0], 3), dtype=np.float32)
    pcl[:, 0] = (xs - cx) * depth / fx
    pcl[:, 1] = (ys - cy) * depth / fy
    pcl[:, 2] = depth
    
    return pcl

def predict_with_bbox(bboxes: sv.Detections, predictor: Predictor, img: Image) -> List[dict]:
    """Retrive sam_result as a list of dict to convert to sv.Detection for visualization.

    Args:
        bboxes (sv.Detections): predicted bboxes from groundingdino
        predictor (Predictor): nanosam predictor class
        img (Image): PIL.Image

    Returns:
        sam_result (List[dict]):
        dict = {
            'area': int(binary_mask.sum().item()),
            'bbox': bboxes[i].xyxy[0],
            'segmentation': binary_mask.squeeze(0).cpu().numpy(),
        }
    """
    predictor.set_image(img)
    sam_result = []

    for i in range(len(bboxes)):
        points, point_labels = bbox2points(bboxes[i].xyxy[0])
        mask, mask_iou, _ = predictor.predict(points, point_labels)
        # Compute binary mask using highest iou score mask
        max_iou, idx = torch.max(mask_iou, dim=-1) # expect mask_iou is (1, 4) dim tensor
        binary_mask = (mask[0, idx] > 0) # grab first batch with highest iou score mask 
        result = {
            'area': int(binary_mask.sum().item()),
            'bbox': bboxes[i].xyxy[0],
            'segmentation': binary_mask.squeeze(0).cpu().numpy(),
        }
        sam_result.append(result)
    return sam_result

def predict_and_annotate(model: Model, predictor: Predictor, img_path: str, caption: str) -> Tuple[List[str], List[sv.Detections], List[dict]]:
    """Retrive a tuple of detected labels, bboxes, and sam_result and visualize result.
    

    Args:
        model (Model): GroundingDINO model
        predictor (Predictor): nanosam predictor
        img_path (str): path to image . I might change in future when camera is used.
        caption (str): target objects to be detected

    Returns:
        labels (List[str]): detected labels
        bboxes (List[sv.Detection]): detected bounding box from groundingdino
        sam_result (List[dict]): 
        dict = {
            'area': int(binary_mask.sum().item()),
            'bbox': bboxes[i].xyxy[0],
            'segmentation': binary_mask.squeeze(0).cpu().numpy(),
        }
    """
    # Detect bounding box with text
    pil_img = Image.open(img_path).convert("RGB")
    img = np.array(pil_img)
    bboxes, labels = model.predict_with_caption(
        image=img,
        caption=caption
    )

    # Detect mask with bounding box
    sam_result = predict_with_bbox(bboxes=bboxes, predictor=predictor, img=pil_img)
    masks = sv.Detections.from_sam(sam_result=sam_result)
    annotate(pil_img, bboxes=bboxes, labels=labels, masks=masks)
    plt.imshow(pil_img)
    
    return labels, bboxes, sam_result

def annotate(image, bboxes: sv.Detections, labels: List[str], masks: sv.Detections):
    """This function overwrite result on image."""
    customlabels = [f'{detected_obj} {confidence:2f}'
                for detected_obj, confidence in zip(labels, bboxes.confidence)]

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_img = box_annotator.annotate(scene=image, detections=bboxes)
    annotated_img = label_annotator.annotate(scene=annotated_img, detections=bboxes, labels=customlabels)
    annotated_img = mask_annotator.annotate(scene=annotated_img, detections=masks)
    return annotated_img

def enforce_right_hand_rule(axes: np.ndarray) -> np.ndarray:
    """Overwrite axes, ensure it follows right-hand rule"""
    z_axis = axes[2]
    z_cross = np.cross(axes[0], axes[1])
    if np.dot(z_cross, z_axis) < 0:
        # Flip the third axis to ensure right-handedness
        axes[2] = -axes[2]
    return axes

def remove_pcl_noise(pcl: np.ndarray, nb_neighbors=20, std_ratio=2.0) -> np.ndarray:
    """Overwrite pcl, denoise it."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcl = np.asarray(cl.points)
    return pcl

def axes2matrix(axes: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Convert axes and centroid to 4x4 homogeneous transformation matrix."""
    return make_transform_matrix(R=axes.T,t=centroid)

def estimate_axes_from_pcl(pcl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate axes and centroid from pcl."""
    pca = PCA(n_components=3)
    pca.fit(remove_pcl_noise(pcl))
    axes = enforce_right_hand_rule(pca.components_)
    centroid = np.mean(pcl, axis=0)
    return axes, centroid

def draw_and_show_axes(pcl_list: List, axes_list: List, axis_length: float=0.05):
    """Show axes on pcl, press 'q' to quit."""
    pcd_list = []
    line_set_list = []
    for pcl, axes in zip(pcl_list, axes_list):
        # Create pcl
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd_list.append(pcd)
        
        # Create axes
        lines = []
        colors = [[1,0,0], [0,1,0], [0,0,1]]
        origin = np.mean(pcl, axis=0)
        points = [origin]
        for i in range(3):
            axis_end = origin + axes[i] * axis_length
            points.append(axis_end)
            lines.append([0, i+1])
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set_list.append(line_set)
    # Visualize
    o3d.visualization.draw_geometries(pcd_list + line_set_list)
    
def visualize_pcl(pcl_list: List, colors=None):
    """Example usage:
    visualize_pcls(
        pcl_list,
        colors=[[1,0,0]]
    )
    """
    pcds = []
    for i, pcl in enumerate(pcl_list):
        # --- handle GPU tensors ---
        if isinstance(pcl, torch.Tensor):
            pcl = pcl.detach().cpu().numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        
        if colors is not None:
            pcd.paint_uniform_color(colors[i%len(colors)])
        
        pcds.append(pcd)
    
    o3d.visualization.draw_geometries(pcds)
