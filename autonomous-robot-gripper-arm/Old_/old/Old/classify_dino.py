"""
Grounding DINO - Fixed for latest transformers version

Shows live camera feed + detections side by side

Install:
pip install torch torchvision transformers pillow opencv-python
"""

# ============================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# ============================================================

# Detection prompts - define what objects to search for
# Each prompt can contain multiple objects separated by periods
DETECTION_PROMPTS = {
    'table_objects': 'egg . potato . wooden cube . block . sphere',
    'food_items': 'apple . egg . banana . potato . orange . bread',
    'all_objects': 'object . item . thing',
    'kitchen': 'cup . plate . bowl . spoon . fork . knife . bottle',
    'custom': 'apple . egg . phone . keys . pen'
}

# Set which prompt to use by default
DEFAULT_PROMPT = 'custom'

# Model selection
# Options: 
#   'IDEA-Research/grounding-dino-base' - balanced (recommended)
#   'IDEA-Research/grounding-dino-tiny' - faster but less accurate
MODEL_ID = 'IDEA-Research/grounding-dino-base'

# Detection threshold (0.0 to 1.0)
# Higher = fewer but more confident detections
# Lower = more detections but may include false positives
DETECTION_THRESHOLD = 0.35

# Camera settings
CAMERA_INDEX = 0  # Usually 0 for built-in webcam, 1 for external

# Display settings
SHOW_CONFIDENCE = True  # Show confidence score on labels
BBOX_COLOR = (0, 255, 0)  # Bounding box color (B, G, R) - Green
BBOX_THICKNESS = 2
TEXT_COLOR = (0, 0, 0)  # Label text color (B, G, R) - Black
TEXT_BG_COLOR = (0, 255, 0)  # Label background color - Green

# ============================================================
# END CONFIGURATION
# ============================================================

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ===== SETUP =====
print("Loading Grounding DINO...")

# Use MPS (Metal Performance Shaders) for M3 acceleration
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

print("Model loaded!")

# ===== DETECTION FUNCTION (FIXED) =====
def detect_objects(frame, text_prompt="egg . potato . wooden cube . block"):
    """
    Detect objects using text prompts
    
    Args:
        frame: OpenCV frame (BGR)
        text_prompt: Objects to find (separate with periods)
        
    Returns:
        frame with boxes, list of detections
    """
    # Convert BGR to RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Process inputs
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    # Run detection
    with torch.no_grad():
        outputs = model(**inputs)
    
    # FIXED: Post-process with correct parameters
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[image.size[::-1]],
        threshold=0.35  # Changed from box_threshold
    )[0]
    
    # Draw results
    result_frame = frame.copy()
    detections = []
    
    # Check if we have boxes
    if len(results["boxes"]) > 0:
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            # Filter by score manually
            if float(score) < 0.35:
                continue
                
            x1, y1, x2, y2 = map(int, box.tolist())
            confidence = float(score)
            
            detections.append({
                'label': label,
                'box': (x1, y1, x2, y2),
                'confidence': confidence
            })
            
            # Visualize
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label}: {confidence:.2f}"
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(result_frame, (x1, y1-text_height-10), 
                         (x1+text_width, y1), (0, 255, 0), -1)
            
            cv2.putText(result_frame, label_text, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result_frame, detections


# ===== MAIN LOOP =====
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot access camera!")
    print("On macOS: System Settings → Privacy & Security → Camera")
    exit()

print("\nGrounding DINO Detection (M3 Pro Optimized)")
print("\nAvailable detection modes:")
for i, (name, prompt) in enumerate(DETECTION_PROMPTS.items(), 1):
    print(f"  '{i}' - {name}: {prompt}")
print("\nOther controls:")
print("  'q' - Quit")
print("  's' - Save frame")
print("  'space' - Pause/Resume detection")

search_mode = DETECTION_PROMPTS[DEFAULT_PROMPT]
current_mode_name = DEFAULT_PROMPT
paused = False
last_result = None
last_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Always show live camera feed
    display_frame = frame.copy()
    
    # Run detection unless paused
    if not paused:
        result_frame, detections = detect_objects(frame, search_mode)
        last_result = result_frame
        last_detections = detections
    else:
        result_frame = last_result if last_result is not None else frame
        detections = last_detections
    
    # Add info overlay to camera feed
    cv2.putText(display_frame, "LIVE CAMERA", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if paused:
        cv2.putText(display_frame, "PAUSED", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add info overlay to detection result
    cv2.putText(result_frame, f"DETECTIONS: {len(detections)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_frame, f"Mode: {current_mode_name}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(result_frame, f"Search: {search_mode[:35]}", (10, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Show both windows
    cv2.imshow('Live Camera Feed', display_frame)
    cv2.imshow('Grounding DINO Detection', result_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("camera_feed.jpg", display_frame)
        cv2.imwrite("detected.jpg", result_frame)
        print(f"\n📸 Saved both frames! Detections:")
        for det in detections:
            print(f"  {det['label']}: {det['confidence']:.2f}")
    elif key == ord(' '):
        paused = not paused
        print(f"Detection {'paused' if paused else 'resumed'}")
    elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
        # Map number to prompt
        mode_list = list(DETECTION_PROMPTS.items())
        mode_idx = key - ord('1')
        if mode_idx < len(mode_list):
            current_mode_name, search_mode = mode_list[mode_idx]
            print(f"\nSwitched to: {current_mode_name}")
            print(f"Searching for: {search_mode}")

cap.release()
cv2.destroyAllWindows()


# ===== TIPS =====
"""
Text prompt syntax:
- Separate with periods: "egg . potato . cube"
- Be specific: "white egg . brown potato . wooden block"
- Use synonyms: "sphere . ball . round object"

Performance on M3 Pro:
- Expected: 5-15 FPS depending on image size
- Press space to pause detection and keep last result
- Camera feed runs smoothly even when detection is processing

Threshold tuning:
- Change threshold=0.35 to higher (0.5) for fewer but more confident detections
- Change to lower (0.25) for more detections with lower confidence
"""