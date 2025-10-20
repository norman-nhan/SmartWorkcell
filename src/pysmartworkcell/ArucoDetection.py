from typing import List, Tuple
import cv2
import cv2.aruco as aruco
import numpy as np
from pysmartworkcell.calibration_utils import vectors2matrix

class ArucoDetectionNode:
    def __init__(self, dictionary: int, cam_matrix: np.ndarray, dist_coeffs: np.ndarray, parameters=None, marker_length: float=0.1):        
        # Tuning parameters
        self.parameters = parameters if parameters is not None else aruco.DetectorParameters()
        self.parameters.minMarkerPerimeterRate = 0.05
        self.parameters.maxMarkerPerimeterRate = 3.0
        self.parameters.polygonalApproxAccuracyRate = 0.02
        self.parameters.minCornerDistanceRate = 0.1
        self.parameters.minMarkerDistanceRate = 0.05
        self.parameters.errorCorrectionRate = 0.3
        self.parameters.maxErroneousBitsInBorderRate = 0.02
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.dictionary = aruco.getPredefinedDictionary(dictionary)
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)
        self.cam_matrix = cam_matrix 
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
    
    def estimatePoseSingleMarkers(self, corners):
        # Prepare 3D object points
        half_length = self.marker_length/2
        objp = np.array([
            [-half_length,  half_length, 0],
            [ half_length,  half_length, 0],
            [ half_length, -half_length, 0],
            [-half_length, -half_length, 0],
        ], dtype=np.float32)

        rvecs, tvecs = [], []
        for corner in corners:
            success, rvec, tvec = cv2.solvePnP(
                objp, corner[0], self.cam_matrix, self.dist_coeffs,
            )
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
        return rvecs, tvecs        

    def estimate_maker_pose_from_frame(self, frame) -> Tuple[bool, List[int], List[np.ndarray]]:
        """Estimate marker pose from a single frame.
        
        Returns
        -------
        marker_found: bool
            Whether any markers were detected in the frame.
        ids: List[int]
            List of marker ids.
        T_list: List[np.ndarray]
            List of transformation matrices.
        
        Parameters
        ----------
        frame: np.ndarray
            Color image.
        """
        marker_found = False
        T_list = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None:
            marker_found = True
            rvecs, tvecs = self.estimatePoseSingleMarkers(corners)
            for rvec, tvec in zip(rvecs, tvecs):
                axis_length = min(0.02, self.marker_length / 2.0)
                cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, rvec, tvec, axis_length)
                T = vectors2matrix(rvec=rvec, tvec=tvec)
                T_list.append(T)

        return marker_found, ids.flatten().tolist(), T_list
    
