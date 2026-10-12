import cv2
import numpy as np
import glob
import os

# ---------------- SETTINGS ----------------
CHESSBOARD_SIZE = (9, 6)     # INNER corners
SQUARE_SIZE = 0.0267         # meters (3.2 cm)
IMAGE_DIR = "/home/samdrishti/codes/stereo_captures"
IMAGE_SIZE = (640, 480)

# ---------------- PREPARE OBJECT POINTS ----------------
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                       0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints = []
imgpointsL = []
imgpointsR = []

# ---------------- LOAD IMAGE PAIRS ----------------
left_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "L_*.jpg")))
print("Total Left Images Found:", len(left_images))

for left_path in left_images:
    file_id = left_path.split("_")[-1]       # 0001.jpg
    right_path = os.path.join(IMAGE_DIR, "R_" + file_id)

    if not os.path.exists(right_path):
        continue

    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # 🔹 CHANGE: Removed histogram equalization
    # grayL = cv2.equalizeHist(grayL)
    # grayR = cv2.equalizeHist(grayR)

    retL, cornersL = cv2.findChessboardCornersSB(
        grayL, CHESSBOARD_SIZE,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    retR, cornersR = cv2.findChessboardCornersSB(
        grayR, CHESSBOARD_SIZE,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )

    if retL and not retR:
        print(f"This left image is ok but right failed: {left_path}")
    if not retL and retR:
        print(f"This right image is ok but left failed: {right_path}")
    if not retL and not retR:
        print(f"This pair is bad: {left_path}")

    # 🔹 CHANGE: Ensure shape matches before appending
    if retL and retR and cornersL.shape == cornersR.shape:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

print("Valid Pairs Used:", len(objpoints))

# ---------------- CALIBRATE INDIVIDUAL CAMERAS ----------------
retL, mtxL, distL, _, _ = cv2.calibrateCamera(
    objpoints, imgpointsL, IMAGE_SIZE, None, None
)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(
    objpoints, imgpointsR, IMAGE_SIZE, None, None
)

# ---------------- STEREO CALIBRATION ----------------
# 🔹 CHANGE: Added CALIB_USE_INTRINSIC_GUESS for better stability
stereo_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS

# 🔹 CHANGE: Stricter termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7)

retStereo, camMatrixL, distL, camMatrixR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    mtxL,
    distL,
    mtxR,
    distR,
    IMAGE_SIZE,
    criteria=criteria_stereo,
    flags=stereo_flags
)

# ---------------- RECTIFICATION ----------------
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    camMatrixL, distL,
    camMatrixR, distR,
    IMAGE_SIZE,
    R, T
)

mapLx, mapLy = cv2.initUndistortRectifyMap(
    camMatrixL, distL, R1, P1,
    IMAGE_SIZE, cv2.CV_32FC1
)
mapRx, mapRy = cv2.initUndistortRectifyMap(
    camMatrixR, distR, R2, P2,
    IMAGE_SIZE, cv2.CV_32FC1
)

# ---------------- PRINT RMS ERRORS ----------------
print("\n========== CALIBRATION RESULTS ==========")
print(f"Left Camera RMS Error  : {retL}")
print(f"Right Camera RMS Error : {retR}")
print(f"Stereo RMS Error       : {retStereo}")

# 🔹 CHANGE: Print baseline to verify IMX219 module
baseline = np.linalg.norm(T)
print(f"Stereo Baseline (meters): {baseline:.4f}")
print("=========================================\n")

# ---------------- SAVE TO XML ----------------
fs = cv2.FileStorage("stereo_calibration_4.xml", cv2.FILE_STORAGE_WRITE)

fs.write("camMatrixL", camMatrixL)
fs.write("distL", distL)
fs.write("camMatrixR", camMatrixR)
fs.write("distR", distR)
fs.write("R", R)
fs.write("T", T)
fs.write("E", E)
fs.write("F", F)
fs.write("Q", Q)
fs.write("mapLx", mapLx)
fs.write("mapLy", mapLy)
fs.write("mapRx", mapRx)
fs.write("mapRy", mapRy)

fs.release()

print("Stereo Calibration Successful!")
print("File saved as stereo_calibration_2.xml")