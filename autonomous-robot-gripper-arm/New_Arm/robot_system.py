from pathlib import Path
import numpy as np, time

from Communication.communication import RobotInterface, find_arduino_uno
from Communication.angles_to_pwm import convert_to_pwm
from Control.Kinematics.Ffnn_IK.ik import Ik_hand
from Control.Move.move import move_servos
from Perception.Camera.camera import Camera
from Perception.Object_Detection.classify_yolo import setup_model, classify
from Control.Kinematics.fk import arm_end_to_base_transform_matrix, camera_to_arm_end_transform_matrix
from Verbose_Functions.testing_utils import visualize_frame, print_detection_summary


class RobotSystem:
    def __init__(self, params):
        # Save and unpack params
        self.params = params
        self.prompt = params.prompt
        self.init_angles = params.init_angles
        self.camera_device_index = params.camera_device_index
        self.verbose = params.verbose
        self.testing_no_arduino = params.testing_no_arduino
        self.nr_pictures = params.nr_pictures_when_searching
        self.z_top_object = params.z_top_object
        self.pwm_releasing_shorter_time_than_grip = params.pwm_releasing_shorter_time_than_grip
        self.pwm_release_delta = params.pwm_release_delta


        self.delay_between_moves = params.delay_between_moves
        
        self.box_mode_align = params.box_mode_align
        self.box_mode_search_move = params.box_mode_search_move
        self.init_angles_sleep = params.init_angles_sleep
        self.pwm_gripping_delta = params.pwm_gripping_delta
        self.nr_pictures_align = params.nr_pictures_align
        # Communication
        self.port = find_arduino_uno()
        self.interface = RobotInterface(self.port)

        # IK
        self.ik_model_path = Path("Control") / "Kinematics" / "Ffnn_IK" / "best_model_hand.pt"
        self.ik_hand = Ik_hand(self.ik_model_path)

        # Camera
        self.calib_path = Path("Perception") / "Camera" / "CameraCalibration" / "camera_calibration.npz"
        self.camera = Camera(self.calib_path, device_index=self.camera_device_index)

        # Object detection
        self.detection_model_path = Path("Perception") / "Object_Detection" / "yoloe-11l-seg.pt"
        self.model = setup_model(self.prompt, self.detection_model_path)

        self.init_position()


    def init_position(self): 
        # Init arm
        self.angles = self.init_angles
        if not self.testing_no_arduino:
            self._send_angles(self.angles, self.delay_between_moves)
            time.sleep(self.init_angles_sleep)

    ####### 2. Search Objects ############################################
    def search_object(self):
        object_pos = np.array([0.0, 0.0, self.z_top_object])
        found_object = False
        x_centers, y_centers, z_centers = [], [], []
        nr_of_detections = 0
        for i in range(self.nr_pictures):
            frame = self.camera.get_frame()
            detections, _ = classify(frame, self.model)
            if detections and len(self.prompt) == 1:
                obj_pos = self._get_object_position_from_detection(self.angles, detections, self.box_mode_search_move)
                if obj_pos is not None:
                    object_pos[:2] += obj_pos[:2]
                    found_object = True
                    nr_of_detections += 1 # CHANGED HERE NOT TESTED
                if self.verbose:
                    x_center, y_center = visualize_frame(frame, detections, i + 1, self.nr_pictures, self.prompt, self.z_top_object)
                    if x_center is not None:
                        x_centers.append(x_center)
                        y_centers.append(y_center)
                        z_centers.append(self.z_top_object)
            else:
                print("CANT FIND OBJECT OR Program assumes only one object in prompt.")

        if self.verbose:
            _ = print_detection_summary(x_centers, y_centers, z_centers, detections_count=len(x_centers))
        
        if found_object : #CHANGED HERE NOT TESTED
                object_pos[:2] /= nr_of_detections # CHANGED HERE NOT TESTED
        if not found_object:
            raise RuntimeError("Could not find object.")
        
        if self.verbose and len(x_centers) > 0:
            mean_x, mean_y = np.mean(x_centers), np.mean(y_centers)
            cam_coords = self._pixel_to_camera_xy(mean_x, mean_y)
            print("hej\n")
            if cam_coords is not None:
                print(f"Camera coords (avg): x={cam_coords[0]:.3f}, y={cam_coords[1]:.3f}, z={cam_coords[2]:.3f}")

        return object_pos

    ####### 3. Move Above Object ############################################
    ####### 5. Step Down Until Gripping Height #################################
    ####### 7. Move To Drop-Off Point ############################################
    def move_servos(self, position, delay):
        move_servos(self.interface, self.ik_hand, self._convert_to_pwm, position, delay, self.testing_no_arduino)



    ####### 4. Align Hand To Object ############################################
    def alignXY(self, hand_pos, object_pos):
        for _ in range(self.nr_pictures_align): 
            frame = self.camera.get_frame()
            detections, _o = classify(frame, self.model)
            if detections:
                break

        if not detections: 
            raise RuntimeError(f"\nNo detection in alignement!, tried taking {self.nr_pictures} pictures. \n")
            
        angles = self.ik_hand.get_angles(*hand_pos)
        object_pos = self._get_object_position_from_detection(angles, detections, self.box_mode_align)

        if object_pos is None:
            raise RuntimeError("Ray pointing up ")

        hand_pos[0] = object_pos[0]
        hand_pos[1] = object_pos[1]

        if self.verbose:
            print(f"\n object_pos: x: {object_pos[0]}, y : {object_pos[1]}\n")
            print("######## ALIGNMENT COMPLETE ########\n")

        return hand_pos
    





    ####### 6. Grip Object #######################################################
    ####### 8. Release Object #######################################################
    def grip_or_release(self, grip):
        if self.testing_no_arduino:
            print("[TEST NO ARDUINO] Skipping servo grip/release commands.")
            return
        print("\n\tGRIPPING OR RELEASING\n")
        delta = self.pwm_gripping_delta if grip else -self.pwm_release_delta 
        self.interface.move_servo(servo_id=4, pwm=1500 + delta)

        sleep_time = self.params.delay_gripping if grip else self.params.delay_gripping - self.pwm_releasing_shorter_time_than_grip
        time.sleep(sleep_time)
        self.interface.move_servo(servo_id=4, pwm=1500)



    ############################################################################
    ############################################################################
    ####### HELPER METHODS #######################################################
    ############################################################################
    ############################################################################

    
    
    def _get_detection_box_center(self, detection):
        xyxy = detection["xyxy"]
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = (xyxy[1] + xyxy[3]) / 2
        return x_center, y_center

    def _get_detection_box_bottom_middle(self, detection):
        xyxy = detection["xyxy"]
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_bottom = xyxy[3]
        return x_center, y_bottom

    def _get_object_position_from_detection(self, angles, detections, box_mode):
        if not detections:
            print("No detections found.")
            return None

        if box_mode == "center":
            x, y = self._get_detection_box_center(detections[0])
        else:
            x, y = self._get_detection_box_bottom_middle(detections[0])

        coords_xy = self._pixel_to_world_xy(x, y, angles)
        if coords_xy is None:
            return None
        return np.array([coords_xy[0], coords_xy[1], self.z_top_object])


    def _pixel_to_world_xy(self, x_center, y_center, angles):
        C = camera_to_arm_end_transform_matrix()
        A = arm_end_to_base_transform_matrix(*angles)
        T = A @ C

        #INCLUDED Z_PLANE!
        z_plane = self.z_top_object if self.box_mode_search_move == "center" else 0.0
        if self.verbose:
            print("\nZ height for intersection plane for ray method: ", z_plane)
        coords = self.camera.pixel_to_world_coords_floor(x_center, y_center, T, z_plane=z_plane)


        if coords is None: 
            print("WARNING Ray towards object not pointing down towards table!")
            return None
        return coords[:2]
    
  
    def _convert_to_pwm(self, servo_id, angle_radians):
        return convert_to_pwm(servo_id, angle_radians)

    def _send_angles(self, angles, delay):
        self.angles = angles
        for i, a in enumerate(angles):
            pwm = self._convert_to_pwm(i, a)
            self.interface.move_servo(i, pwm)
            time.sleep(delay)


    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    # may want to remove :
    def _pixel_to_world_xy_with_known_depth(self, u, v, depth, angles):
        C = camera_to_arm_end_transform_matrix()
        A = arm_end_to_base_transform_matrix(*angles)
        T_cw = A @ C
        pt_cam = self.camera.pixel_to_camera_coords(u, v, depth)
        pt_world = (T_cw @ pt_cam)[:3]
        return pt_world[:2]

    def _pixel_to_camera_xy(self, x_center, y_center):
       
        C = camera_to_arm_end_transform_matrix()
        A = arm_end_to_base_transform_matrix(*self.angles)
        T_cw = A @ C

        z_plane = self.z_top_object if self.box_mode_search_move == "center" else 0.0
        pt_world = self.camera.pixel_to_world_coords_floor(x_center, y_center, T_cw, z_plane)
        if pt_world is None:
            print("WARNING Ray towards object not pointing down towards table!")
            return None
        
        pt_cam = np.linalg.inv(T_cw) @ np.append(pt_world, 1)
        return pt_cam[:3]  


    def _get_object_position_known_size(self, detection, object_size_m, mode="height"):
        """
        Estimate object position in the camera's coordinate frame
        using known real-world size (height or width).
        """
        xyxy = detection["xyxy"]
        x_min, y_min, x_max, y_max = xyxy

        u = (x_min + x_max) / 2
        v = (y_min + y_max) / 2
        
        C = camera_to_arm_end_transform_matrix()
        A = arm_end_to_base_transform_matrix(*self.angles)
        T = A @ C
        
        coords = self.camera.pixel_to_world_coords_known_size(
            u=u,
            v=v,
            min_px=y_min if mode == "height" else x_min,
            max_px=y_max if mode == "height" else x_max,
            object_size_m=object_size_m,
            T_cw=T,
            mode=mode
        )
        return coords
