import cv2
import cv2.aruco as aruco
import numpy as np

class ArucoDetectionNode:
    def __init__(self, dictionary=cv2.aruco.DICT_4X4_50, parameters=None, cam_matrix=None, dist_coeffs=None):
        self.cap = cv2.VideoCapture(1)
        
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        self.parameters = parameters if parameters is not None else cv2.aruco.DetectorParameters()
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = 0.05 # in meters
        # create detector
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def estimate_marker_pose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        if ids is not None:
            print(f'corners: {corners}, ids: {ids}')
            aruco.drawDetectedMarkers(frame, corners, ids)
        
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.cam_matrix, self.dist_coeffs
        )
        
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, rvec, tvec, self.marker_length/2)

            # Print pose
            print("rvec:", rvec)
            print("tvec:", tvec)

    def execute(self):
        try:
            count = 0
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to grab frame")
                    break
                
                self.estimate_marker_pose(frame)

                cv2.imshow("Camera ON", frame)

                # Press 'q' to quit
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                if key == ord('s'):
                    cv2.imwrite(f'/calibration_images/{count}.png', frame)
                    count+=1
                    print(f'[INFO] Saved {count}.png')

        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    calibrated = np.load("camera_calib.npz")
    node = ArucoDetectionNode(cam_matrix=calibrated['camera_matrix'], dist_coeffs=calibrated['dist_coeffs'])
    node.execute()

if __name__ == "__main__":
    main()