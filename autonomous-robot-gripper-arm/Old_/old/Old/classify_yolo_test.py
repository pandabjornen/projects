import torch
torch.set_default_dtype(torch.float32)  # MPS float64 workaround
import cv2
import time
from ultralytics import YOLOE

# ===== PARAMETERS ==============================================================================================================
MODEL_PATH = "yoloe-11l-seg.pt"  # small: yoloe-11s-seg.pt, medium: yoloe-11m-seg.pt, large: yoloe-11l-seg.pt"
PROMPT = ["pen", "iphone", "egg", "apple"]  
TARGET_FPS = 10  
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
VIDEO_CAPTURE_INDEX = 0 # 0 for laptop camera

# =========================================================================================================

# Load model
model = YOLOE(MODEL_PATH)



positional_encodings = model.get_text_pe(PROMPT)  # do this on cpu to not get error because of MPS
model.set_classes(PROMPT, positional_encodings) 

model.to("mps")  # for macboook


# windows?: 
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

cap = cv2.VideoCapture(VIDEO_CAPTURE_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

frame_time = 1.0 / TARGET_FPS

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()  # Draws boxes + labels + confidences (default)

    detections = results[0].boxes
    if detections is not None and len(detections) > 0:
        # Overlay detection status text
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