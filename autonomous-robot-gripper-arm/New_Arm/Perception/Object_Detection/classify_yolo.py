import torch
import time
import cv2
from ultralytics import YOLOE

def setup_model(
    prompt=["pen", "iphone", "egg", "apple"],
    model_path="yoloe-11l-seg.pt" 
):
    """
    Loads and setups a YOLO model with positional encodings for the given prompt. 
    Should auto-detect device. 
    
    Args:
        prompt: List of text prompts/classes for open-vocabulary detection.
        model_path: Path to the YOLO model file. # small: yoloe-11s-seg.pt, medium: yoloe-11m-seg.pt, large: yoloe-11l-seg.pt
        
    Returns:
        Configured YOLO model ready for inference.
    """
    model = YOLOE(model_path)
    positional_encodings = model.get_text_pe(prompt)  # do positional encodings on cpu since doesnt work on macbook mps. 
    model.set_classes(prompt, positional_encodings)
    device =  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    print("\n \nYOLO model runs on device: ", device, "\n \n")
    return model

def classify(frame,model):
    """
    Runs YOLO detection on a single frame (NumPy array) and returns list of detections.
    
    Args:
        frame: Input frame as NumPy array (H, W, 3) in BGR format.
        model: Configuered YOLO model.        
    Returns:
        List of dicts: [{"class": str, "conf": float, "xyxy": [x1, y1, x2, y2]}, ...]. conf = confidence
        Empty list if no detections.
    """

    results = model(frame, verbose=False)
    
    detections = []
    if not results or not results[0].boxes:
        return detections, results

    boxes = results[0].boxes
    for box in boxes:
        
        label = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
        label_name = model.names[label]  
        
        detections.append({
            "class": label_name,
            "conf": confidence,
            "xyxy": xyxy
        })
    
    return detections, results


if __name__ == "__main__":
    # ===== PARAMETERS ==============================================================================================================
    MODEL_PATH = "yoloe-11l-seg.pt"  # small: yoloe-11s-seg.pt, medium: yoloe-11m-seg.pt, large: yoloe-11l-seg.pt"
    # PROMPT = ["pen", "iphone", "egg", "apple", "tennis ball"]  
    # PROMPT = ["white circle of tape"]
    PROMPT = ["yellow-green ball "]
    TARGET_FPS = 10  
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    VIDEO_CAPTURE_INDEX = 0 # 0 for laptop camera

    # =========================================================================================================

    


    model = setup_model(PROMPT, MODEL_PATH) 

    cap = cv2.VideoCapture(VIDEO_CAPTURE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_time = 1.0 / TARGET_FPS

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret: break

        detections, results = classify(frame, model=model)
        annotated_frame = results[0].plot()  # Draws boxes + labels + confidences (default)

        print('detections :' , detections)

        if detections:  # Check if any detections
            status_text = "Objects Detected"
            cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            status_text = "No Objects Detected"
            cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(f"YOLOE Multi-Object Detection ({TARGET_FPS} FPS)", annotated_frame)

        elapsed = time.time() - start_time
        time.sleep(max(0, frame_time - elapsed))

        if cv2.waitKey(1) & 0xFF == ord("q"): break #press q to stop

    cap.release()
    cv2.destroyAllWindows()