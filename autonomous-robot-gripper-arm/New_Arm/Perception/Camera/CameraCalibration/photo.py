import cv2
import os
from datetime import datetime

def main():
    # Camera index (0 = first camera; try 1, 2, etc. if multiple are connected)
    cam_index = 1
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"❌ Could not open camera with index {cam_index}")
        return

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create output directory
    save_dir = "photos"
    os.makedirs(save_dir, exist_ok=True)

    print("✅ Camera ready.")
    print("Press [SPACE] to take photo, [ESC] to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        cv2.imshow("USB Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        # SPACE = take photo
        if key == 32:  # spacebar
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"photo_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"📷 Saved {filename}")

        # ESC = exit
        elif key == 27:
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()