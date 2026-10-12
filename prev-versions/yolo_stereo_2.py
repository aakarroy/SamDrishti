import numpy as np
import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

# ==============================
# LOAD YOLO
# ==============================
print("Loading YOLO...")
model = YOLO("/home/samdrishti/codes/yolov8n_ncnn_model", task="detect")

dummy = np.zeros((320,320,3), dtype=np.uint8)
model(dummy)

# ==============================
# LOAD CALIBRATION
# ==============================
print("Loading calibration...")

fs = cv2.FileStorage(
"/home/samdrishti/codes/stereo_calibration_2.xml",
cv2.FILE_STORAGE_READ)

K1 = fs.getNode("camMatrixL").mat()
D1 = fs.getNode("distL").mat()
K2 = fs.getNode("camMatrixR").mat()
D2 = fs.getNode("distR").mat()
R  = fs.getNode("R").mat()
T  = fs.getNode("T").mat()

image_size = (640,480)

R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(
K1,D1,K2,D2,image_size,R,T)

fs.release()

# ==============================
# RECTIFICATION MAPS
# ==============================
mapLx,mapLy = cv2.initUndistortRectifyMap(
K1,D1,R1,P1,image_size,cv2.CV_32FC1)

mapRx,mapRy = cv2.initUndistortRectifyMap(
K2,D2,R2,P2,image_size,cv2.CV_32FC1)

# ==============================
# START CAMERAS
# ==============================
print("Starting cameras...")


camL = Picamera2(0)
camL.configure(camL.create_video_configuration(
main={"size":image_size,"format":"BGR888"}))
camL.start()

camR = Picamera2(1)
camR.configure(camR.create_video_configuration(
main={"size":image_size,"format":"BGR888"}))
camR.start()

# ==============================
# STEREO MATCHER
# ==============================
stereo = cv2.StereoSGBM_create(
minDisparity=0,
numDisparities=64,
blockSize=5,
P1=8*3*5**2,
P2=32*3*5**2
)

# ==============================
# FPS
# ==============================
prev = time.time()
fps = 0

print("Running...")

# ==============================
# MAIN LOOP
# ==============================
try:

    while True:

        frameL = camL.capture_array()
        frameR = camR.capture_array()

        # Rectify
        left = cv2.remap(frameL,mapLx,mapLy,cv2.INTER_LINEAR)
        right = cv2.remap(frameR,mapRx,mapRy,cv2.INTER_LINEAR)

        display = left.copy()

        # =====================
        # YOLO
        # =====================
        results = model(
            left,
            imgsz=320,
            conf=0.25,
            verbose=False
        )

        grayL = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)

        for box in results[0].boxes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            # Expand ROI slightly
            pad = 20
            x1 = max(0,x1-pad)
            y1 = max(0,y1-pad)
            x2 = min(image_size[0],x2+pad)
            y2 = min(image_size[1],y2+pad)

            roiL = grayL[y1:y2,x1:x2]
            roiR = grayR[y1:y2,x1:x2]
            
            h, w = roiL.shape
            if w < 80 or h < 80:
                continue

            disp = stereo.compute(roiL,roiR).astype(np.float32)/16.0

            valid = disp[disp > 0]

            if len(valid) == 0:
                continue

            disparity = np.median(valid)

            baseline = abs(T[0][0])
            focal = K1[0,0]

            depth = float((focal * baseline) / disparity)

            cls = int(box.cls[0])

            label = f"{model.names[cls]} {depth:.2f}m"
            cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(display,
                        label,
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2)

        # FPS
        now = time.time()
        fps = 0.9*fps + 0.1*(1/(now-prev))
        prev = now

        cv2.putText(display,
                    f"FPS:{fps:.2f}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2)

        cv2.imshow("SamDrishti",display)

        if cv2.waitKey(1)==ord('q'):
            break

finally:

    camL.stop()
    camR.stop()

    camL.close()
    camR.close()

    cv2.destroyAllWindows()