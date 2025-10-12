import cv2
from SmartWorkcell.calibration_utils import (
    get_camera_intrinsic, load_transform_mtx, save_transformation_mtx
)
from mac_ArucoDetection import ArucoDetectionNode
import cv2.aruco as aruco
import argparse

def calibrate(args):
    # Load camera intrinsic
    cam_mtx, dist_coeffs = get_camera_intrinsic(args.intrinsic_path)
    # 1. Open camera and detect aruco markers
    
    node = ArucoDetectionNode(
        args.marker_dict, args.marker_len, cam_mtx, dist_coeffs
    )

    # 2. Load robot pose in marker frame
    T_marker_robot = load_transform_mtx(args.marker2robot)
    T_cam_robot = None
    
    try:
        cap = cv2.VideoCapture(args.camera_id)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read frame")
                break
            success, ids, T_list = node.estimate_marker_poses_from_frame(frame)
            cv2.imshow("Aruco Detection", frame)
            key=cv2.waitKey(1)
            
            # Suppose we only know robot pose in marker ID=0's frame
            if success:
                for id, T in zip(ids, T_list):
                    if id == 0:
                        T_cam_robot = T @ T_marker_robot
                        print("[INFO] T_cam_robot:\n", T_cam_robot)
                        break
                if T_cam_robot is not None:
                    save_transformation_mtx(T_cam_robot, args.save_path)
                    break # stop program
            if key == ord('q'):
                break
    except Exception as e:
        print("[ERROR]", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsic_path', type=str, default="config/mac_calibration.yaml", 
                        help="Path to file contains camera matrix and distortion coefficients")
    parser.add_argument('--marker_dict', type=str, default='DICT_4X4_50')
    parser.add_argument('--marker_len', type=float, default=0.1, 
                        help='Set this parameter correctly to get correct translation vectors')
    parser.add_argument('--marker2robot', type=str, default='config/marker2robot.yaml')
    parser.add_argument('--cam2marker', type=str, default='config/cam2marker.yaml')
    parser.add_argument('--camera_id', type=int, default=1)
    parser.add_argument('-o', '--save_path', default='config/cam2robot.yaml')
    args = parser.parse_args()
    # Convert dict to arucolib dict
    args.marker_dict = getattr(aruco, args.marker_dict) 
    calibrate(args)