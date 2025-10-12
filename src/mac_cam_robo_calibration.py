import cv2
from SmartWorkcell.calibration_utils import get_camera_intrinsic, load_transform_mtx
from mac_ArucoDetection import ArucoDetectionNode
import cv2.aruco as aruco

# =============
# MARKER LENGTH
# =============
# For TRANSLATION VECTORS TO BE CORRECT THIS MUST BE SET CORRECTLY
# If you only want to detect the markers or just care about the orientation, this params is not important.
MARKER_LENGTH = 0.1 # in meters
MARKER_DICT = aruco.DICT_4X4_50
CAMERA_INTRINSIC = "config/mac_calibration.yaml"
MARKERS_TO_ROBOT = "config/markers_to_robot.yaml" # contains T_marker_robot for each marker id


def main():
    # Load camera intrinsic
    cam_mtx, dist_coeffs = get_camera_intrinsic(CAMERA_INTRINSIC)
    # 1. Open camera and detect aruco markers
    node = ArucoDetectionNode(
        MARKER_DICT, MARKER_LENGTH, cam_mtx, dist_coeffs
    )

    # 2. Load robot pose in marker frame
    T_marker_robot = load_transform_mtx(MARKERS_TO_ROBOT)
    T_cam_robot = None
    
    try:
        cap = cv2.VideoCapture(1)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read frame")
                break
            success, ids, T_list = node.estimate_marker_poses_from_frame(frame)
            cv2.imshow("Aruco Detection", frame)
            key=cv2.waitKey(1)
            
            # Suppose we only know robot pose in marker ID=0 frame
            if success:
                for id, T in zip(ids, T_list):
                    if id == 0:
                        T_cam_robot = T @ T_marker_robot
                        print("[INFO] T_cam_robot:\n", T_cam_robot)
                        break
                if T_cam_robot is not None:
                    break # stop program
            if key == ord('q'):
                break
    except Exception as e:
        print("[ERROR]", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()