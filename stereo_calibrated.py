# stereo_calibrate.py
import cv2
import numpy as np
import glob

# -------- CHESSBOARD PARAMETERS ----------
CHESSBOARD = (9, 6)  # inner corners
SQUARE_SIZE = 2.5    # cm (or whatever you used)
# ----------------------------------------

# Prepare object points (3D points in chessboard coordinate system)
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpointsL = []
imgpointsR = []

left_images = sorted(glob.glob("stereo_images/left_*.png"))
right_images = sorted(glob.glob("stereo_images/right_*.png"))

for left_path, right_path in zip(left_images, right_images):
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD, None)

    if retL and retR:
        objpoints.append(objp)

        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), term)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), term)

        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

print("Found", len(objpoints), "valid image pairs")

# ---- INDIVIDUAL CAMERA CALIBRATION ----
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpointsL, grayL.shape[::-1], None, None)

retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpointsR, grayR.shape[::-1], None, None)

# ---- STEREO CALIBRATION ----
flags = (cv2.CALIB_FIX_INTRINSIC)

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    mtxL,
    distL,
    mtxR,
    distR,
    grayL.shape[::-1],
    criteria=criteria,
    flags=flags
)

print("Stereo Calibration RMS error:", retval)
print("Translation vector T:\n", T)
print("Rotation matrix R:\n", R)

# ---- RECTIFICATION ----
RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
    mtxL, distL,
    mtxR, distR,
    grayL.shape[::-1],
    R, T,
    alpha=0
)

# Create rectification maps
mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, RL, PL, grayL.shape[::-1], cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, RR, PR, grayR.shape[::-1], cv2.CV_32FC1)

# ---- SAVE EVERYTHING ----
np.savez("stereo_params.npz",
         mtxL=mtxL, distL=distL,
         mtxR=mtxR, distR=distR,
         R=R, T=T,
         RL=RL, RR=RR, PL=PL, PR=PR,
         Q=Q,
         mapLx=mapLx, mapLy=mapLy,
         mapRx=mapRx, mapRy=mapRy)

print("Stereo parameters saved to stereo_params.npz")
