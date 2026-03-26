import numpy as np


def align(camera_to_arm_end_transform_matrix, arm_end_to_base_transform_matrix,  z_top_of_object, camera, classify, object_detection_model, robot, \
          ik, convert_to_pwm, move_servos,   hand_pos, object_pos ,TESTING_NO_ARDUINO, \
                            tol=0.01, step=0.001, delay_between_moves=0.2, max_iterations = 1000) \
                            -> list[float, float, float]:
    
    """
    assumes above (in z-direction) the object and alignement 
    only needs to happen in global xy plane. 
    

    currently takes new frame everytime adjusting, maybe not needed
    """
    def get_obj_pos_from_detections(hand_pos, camera, detections, z_top_of_object): 

        object_pos = np.array([0.0, 0.0, 0.0])
        
        if detections: 
            xyxy = detections[0]["xyxy"]
            x_min, y_min, x_max, y_max = xyxy[0], xyxy[1], xyxy[2], xyxy[3] 
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            print("\n",x_center, y_center, "\n")
            angles = ik.get_angles(*hand_pos)

            print("\n angles: ",angles, "\n")

            C = camera_to_arm_end_transform_matrix()
            A = arm_end_to_base_transform_matrix(*angles)
            T = A @ C

            coords = camera.pixel_to_world_coords_floor(x_center, y_center, T)

            print(coords)
            print(coords.shape)
            x_center_world = coords[0]
            y_center_world = coords[1]        
            object_pos[0], object_pos[1], object_pos[2] = x_center_world, y_center_world, z_top_of_object
        else: 
            print("Could not find object after moving to above object! starting alignment anyway. ")
        
        return object_pos
    
    frame = camera.get_frame()
    detections, _ = classify(frame, object_detection_model)

    object_pos = get_obj_pos_from_detections(hand_pos, camera, detections, z_top_of_object)

    
    dx = object_pos[0] - hand_pos[0]
    dy = object_pos[1] - hand_pos[1]

    iterations = 0
    while abs(dx) > tol or abs(dy) > tol:
        move_x = step if dx > 0 else -step if abs(dx) > tol else 0
        move_y = step if dy > 0 else -step if abs(dy) > tol else 0
        hand_pos[0] += move_x
        hand_pos[1] += move_y
        
        move_servos(robot, ik, convert_to_pwm, hand_pos, delay_between_moves, TESTING_NO_ARDUINO)
        
        frame = camera.get_frame()
        detections, _ = classify(frame, object_detection_model)    
        object_pos = get_obj_pos_from_detections(hand_pos, camera, detections, z_top_of_object)

        dx = object_pos[0] - hand_pos[0]
        dy = object_pos[1] - hand_pos[1]

        if iterations > max_iterations: 
            return hand_pos
        iterations += 1

    return hand_pos
