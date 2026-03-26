import cv2
import numpy as np

# ===== PARAMETERS =====
CAMERA_INDEX = 0
CANNY_LOW = 50
CANNY_HIGH = 150
MIN_AREA = 500
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_COLOR_COLOR = (0, 255, 0)
CONTOUR_COLOR_COLOR = (0, 255, 0)
BOX_COLOR_COLOR = (255, 0, 0)
TEXT_COLOR_GRAY = (255, 255, 255)
CONTOUR_COLOR_GRAY = (255, 255, 255)
BOX_COLOR_GRAY = (255, 255, 255)
LINE_THICKNESS = 2

# ===== MAIN CODE =====
cap = cv2.VideoCapture(CAMERA_INDEX)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Copies for drawing
    color_result = frame.copy()
    gray_result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Draw on color frame
            cv2.drawContours(color_result, [cnt], -1, CONTOUR_COLOR_COLOR, LINE_THICKNESS)
            cv2.rectangle(color_result, (x, y), (x+w, y+h), BOX_COLOR_COLOR, LINE_THICKNESS)
            cv2.putText(color_result, 'Object', (x, y-10), FONT, TEXT_SCALE, TEXT_COLOR_COLOR, LINE_THICKNESS)
            
            # Draw on grayscale frame
            cv2.drawContours(gray_result, [cnt], -1, CONTOUR_COLOR_GRAY, LINE_THICKNESS)
            cv2.rectangle(gray_result, (x, y), (x+w, y+h), BOX_COLOR_GRAY, LINE_THICKNESS)
            cv2.putText(gray_result, 'Object', (x, y-10), FONT, TEXT_SCALE, TEXT_COLOR_GRAY, LINE_THICKNESS)
    
    cv2.imshow('Detection Color', color_result)
    cv2.imshow('Detection Grayscale', gray_result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
