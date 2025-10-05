import cv2
import numpy as np
import glob
import yaml
from pathlib import Path

class ChessboardCalibration():
    def __init__(self, checkboard_size=(7,9), image_width=640, image_height=480):
        self.checkboard_size = checkboard_size # Chessboard dimensions (inner corners per row & column)
        self.image_w = image_width
        self.image_h = image_height
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Criteria for corner refinement

    # def prepapre_objpoints(self):
        # Prepare 3D points in real-world space (z=0 plane)
        chessboard_w, chessboard_h = self.checkboard_size
        self.empty_objp = np.zeros((chessboard_w*chessboard_h, 3), np.float32)
        self.empty_objp[:, :2] = np.mgrid[0:chessboard_w, 0:chessboard_h].T.reshape(-1, 2)

        # return objp
    def save2yaml(self, camera_matrix, dist_coeffs):
        calibration_data = {
            'camera_matrix': camera_matrix.tolist(), # 3x3
            'dist_coeffs': dist_coeffs.flatten().tolist() # 1x5
        }
        with open('camera_calibration.yaml', 'w') as f:
            # yaml.dump(calibration_data, f, sort_keys=False, default_flow_style=True)
            yaml.dump(
                calibration_data,
                f,
                sort_keys=False,
                default_flow_style=None,  # auto: list stays inline if short
                width=120,                # keeps long lines from wrapping
                indent=2                  # clean indentation for readability
            )
            print(f'[INFO] Successfully saved camera calibration data to camera_calibration.yaml')

    def calibrate(self):
        objpoints = []  # 3D points
        imgpoints = []  # 2D points

        # Load all calibration images (change path)
        images = glob.glob('/Users/tptn/SmartWorkcell/calibration_images/*.png')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            shape = gray.shape[::-1] # (W, H)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkboard_size, None)

            if ret:
                objpoints.append(self.empty_objp)

                # Refine corner locations
                # (11, 11) is the searching kernel size
                # (-1, -1) to disable zerozone 
                corners2 = cv2.cornerSubPix(gray, corners, winSize=(11,11), zeroZone=(-1,-1), criteria=self.criteria)
                imgpoints.append(corners2)

                # Draw & show
                cv2.drawChessboardCorners(img, self.checkboard_size, corners2, ret)
                cv2.imshow('Calibration', img)
                cv2.waitKey(500)
                fname = Path(fname).stem
                success = cv2.imwrite(f'/Users/tptn/SmartWorkcell/detected_chessboard_images/drawn_chessboard_{fname}.png', img)
                if success:
                    print(f'[INFO] Successfully saved detected chessboard to /Users/tptn/SmartWorkcell/detected_chessboard_images/drawn_chessboard_{fname}.png')
                else:
                    print(f'[ERROR] Save path unvalid or other error occurred.')

        cv2.destroyAllWindows()

        # Camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, shape, None, None
        )

        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)

        # Save to file
        np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        self.save2yaml(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

if __name__ == "__main__":
    calibration_node = ChessboardCalibration(checkboard_size=(7,9))
    calibration_node.calibrate()