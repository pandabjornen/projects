import cv2
import numpy as np

def visualize_frame(frame, detections, frame_num, total_frames, PROMPT, Z_TOP_OF_OBJECT):
    temp_frame = frame.copy()
    x_center, y_center = None, None
    if detections and len(PROMPT) == 1:
        xyxy = detections[0]["xyxy"]
        cv2.rectangle(temp_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(temp_frame, "OBJECT DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        x_min, y_min, x_max, y_max = xyxy[0], xyxy[1], xyxy[2], xyxy[3] 
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        cv2.putText(temp_frame, f"Center Point: x={x_center:.2f}, y={y_center:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(temp_frame, "Coords are middle of bbox", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        print(f"Frame {frame_num}: x={x_center:.2f}, y={y_center:.2f}, z={Z_TOP_OF_OBJECT:.2f}")
    else:
        cv2.putText(temp_frame, "NO OBJECT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(temp_frame, f"Picture {frame_num}/{total_frames}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    window_name = f"TESTING: Search Frame {frame_num}"
    cv2.imshow(window_name, temp_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:  # Window closed
            break
    
    cv2.destroyWindow(window_name)
    return x_center, y_center

def print_detection_summary(x_centers, y_centers, z_centers, detections_count):
    if detections_count > 0:
        x_centers = np.array(x_centers)
        y_centers = np.array(y_centers)
        z_centers = np.array(z_centers)
        
        mean_x = np.mean(x_centers)
        mean_y = np.mean(y_centers)
        mean_z = np.mean(z_centers)
        
        std_x = np.std(x_centers)
        std_y = np.std(y_centers)
        std_z = np.std(z_centers)
        
        print(f"\nSummary after {detections_count} detections:")
        print(f"Mean coordinates: x={mean_x:.2f}, y={mean_y:.2f}, z={mean_z:.2f}")
        print(f"Std coordinates: x={std_x:.2f}, y={std_y:.2f}, z={std_z:.2f}")
        
        return np.array([mean_x, mean_y, mean_z])
    return None