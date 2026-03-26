import cv2
import numpy as np
import glob

# Checkerboard settings
CHECKERBOARD = (13, 8)  # number of inner corners per row and column
square_size = 0.03  # 25 mm = 0.025 m


objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size


objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob('photos/*.jpg')  # your calibration image folder

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Optionally visualize
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)
cv2.destroyAllWindows()



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)


print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

np.savez('camera_calibration.npz', mtx=mtx, dist=dist)

data = np.load('camera_calibration.npz')
K = data['mtx']
dist = data['dist']

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)