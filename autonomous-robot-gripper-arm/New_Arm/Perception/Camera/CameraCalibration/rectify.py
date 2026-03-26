import cv2
import numpy as np


calib_file = 'camera_calibration.npz'  # saved with np.savez
data = np.load(calib_file)
K = data['mtx']       # original intrinsics
dist = data['dist']   # distortion coefficients

# Compute new optimal camera matrix for undistortion
# alpha=1 keeps all pixels; can use 0 to crop black borders
frame_width = 1280
frame_height = 720
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (frame_width, frame_height), 1, (frame_width, frame_height))
fx_new, fy_new = newcameramtx[0,0], newcameramtx[1,1]
cx_new, cy_new = newcameramtx[0,2], newcameramtx[1,2]


print("0")
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"❌ Could not open camera with index {1}")
    

print("1")

ret, frame = cap.read()


# Undistort
undistorted = cv2.undistort(frame, K, dist, None, newcameramtx)

cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
