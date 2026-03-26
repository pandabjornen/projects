import cv2
import numpy as np


class Camera:
    def __init__(self, calib_file: str, device_index: int = 0, resolution=(1280, 720), alpha: float = 0.0):
        """
        Camera abstraction for calibrated monocular vision.

        Args:
            calib_file: Path to npz file containing 'mtx' (K) and 'dist'.
            device_index: Camera index (default 0).
        """
        self.device_index = device_index
        self.resolution = resolution

        # Load calibration parameters
        data = np.load(calib_file)
        self.K = data["mtx"]
        self.dist = data["dist"]

        # Compute rectified intrinsics for undistorted frames
        self.new_K, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, resolution, alpha
        )


        # Initialize video capture
        print("initializing camera")
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        print("camera is initialized")


    # --- Frame capture ---
    def get_frame(self):
        """Capture a frame"""
        if not self.cap.isOpened():
            raise RuntimeError("Camera not initialized or not available.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    # --- Undistortion utilities ---
    def undistort_pixel(self, u: float, v: float):
        """
        Undistort a pixel coordinate using calibration parameters.
        Returns a pixel coordinate in the rectified (new_K) image space.
        """
        pts = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pts, self.K, self.dist, P=self.new_K)
        return undistorted[0, 0]  # (u', v')

    # --- Projection math ---
    def pixel_to_camera_coords(self, u: float, v: float, depth: float):
        """
        Convert a pixel (u, v) and depth to 3D coordinates in camera frame.
        """
        
        u, v = self.undistort_pixel(u, v)
        fx, fy = self.new_K[0, 0], self.new_K[1, 1]
        cx, cy = self.new_K[0, 2], self.new_K[1, 2]

        # fx, fy = self.K[0, 0], self.K[1, 1]
        # cx, cy = self.K[0, 2], self.K[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return np.array([x, y, z, 1])



    def pixel_to_world_coords_floor(self, u: float, v: float, T_cw: np.ndarray) -> np.ndarray:
        
        dir_vector_cam = self.pixel_to_camera_coords(u, v, 1.0)
        cam_center_world = T_cw @ np.array([0,0,0,1])
        
        dir_vector_world = (T_cw @ dir_vector_cam) - cam_center_world

        if dir_vector_world[2] < 0:
            return (cam_center_world +  dir_vector_world * -cam_center_world[2] / dir_vector_world[2])[0:3]

        return None
    
    def pixel_to_world_coords_floor(self, u: float, v: float, T_cw: np.ndarray, z_plane: float = 0.0) -> np.ndarray:
        dir_vector_cam = self.pixel_to_camera_coords(u, v, 1.0)
        cam_center_world = T_cw @ np.array([0,0,0,1])
        dir_vector_world = (T_cw @ dir_vector_cam) - cam_center_world

        if dir_vector_world[2] < 0:   #### CHANGED HERE : --> V
            return (cam_center_world + dir_vector_world * (z_plane - cam_center_world[2]) / dir_vector_world[2])[0:3]

        return None



    # --- Lifecycle ---
    def release(self):
        """Release camera resources."""
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()

    ################################################################################################
    ################################################################################################
    ####################################### Known width or height ##################################
    ################################################################################################

    def estimate_distance_with_known_size(self, min_px, max_px, object_size_m, mode="height"):
        """Estimate distance using known object size (height or width)."""
        if mode == "height":
            f = self.K[1, 1]
        elif mode == "width":
            f = self.K[0, 0]
        else:
            raise ValueError("mode must be 'height' or 'width'")
        size_px = abs(max_px - min_px)
        return (f * object_size_m) / size_px


    def pixel_to_world_coords_known_size(self, u, v, min_px, max_px, object_size_m, T_cw, mode="height"):
        """Convert pixel to world coords using known object height or width."""
        d = self.estimate_distance_with_known_size(min_px, max_px, object_size_m, mode)
        pt_cam = self.pixel_to_camera_coords(u, v, d)
        return (T_cw @ pt_cam)[:3]

    

    ################################################################################################
    ################################################################################################
    ################################################################################################
    ################################################################################################






if __name__ == "__main__":

    # Initialize

    # camera = Camera('CameraCalibration\camera_calibration.npz', device_index=1)  # windows
    camera = Camera('CameraCalibration/camera_calibration.npz', device_index=0) # macOS

    
    # print(camera.K)


   
    coords_cam = camera.pixel_to_camera_coords(1200, 700, depth=2.0)
    print("Camera coords:", coords_cam.ravel())

    # Estimate world position assuming the pixel lies on the floor
    import sys
    import os

    #sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from Control.Kinematics.fk import camera_to_arm_end_transform_matrix
    from Control.Kinematics.fk import arm_end_to_base_transform_matrix, arm_transform_points_numpy


    C = camera_to_arm_end_transform_matrix()

    A = arm_end_to_base_transform_matrix(np.deg2rad(0.0),np.deg2rad(30.0),np.deg2rad(30.0), np.deg2rad(-90.0))




    T = A @ C

    print(A)
    
    #640
    #360

    coords_world = camera.pixel_to_world_coords_floor(1280, 360, T)
    print("World coords on floor:", coords_world)
