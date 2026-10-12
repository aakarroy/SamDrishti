import numpy as np
import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

# ===============================
# 1. LOAD YOLO MODEL
# ===============================
print("Loading YOLO model...")
model = YOLO("/home/samdrishti/codes/yolov8n_ncnn_model", task="detect")

# Warmup (important for NCNN speed)
dummy = np.zeros((320,320,3), dtype=np.uint8)
model.predict(dummy, imgsz=320, verbose=False)

# ===============================
# 2. LOAD CALIBRATION
# ===============================
print("Loading Calibration...")
fs = cv2.FileStorage("/home/samdrishti/codes/stereo_calibration_2.xml", cv2.FILE_STORAGE_READ)

K1 = fs.getNode("camMatrixL").mat()
D1 = fs.getNode("distL").mat()
K2 = fs.getNode("camMatrixR").mat()
D2 = fs.getNode("distR").mat()
R  = fs.getNode("R").mat()
T  = fs.getNode("T").mat()

image_size = (640,480)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1,D1,K2,D2,image_size,R,T)
fs.release()

# ===============================
# 3. PRECOMPUTE RECTIFICATION
# ===============================
mapLx,mapLy = cv2.initUndistortRectifyMap(K1,D1,R1,P1,image_size,cv2.CV_32FC1)
mapRx,mapRy = cv2.initUndistortRectifyMap(K2,D2,R2,P2,image_size,cv2.CV_32FC1)

# ===============================
# 4. START CAMERAS
# ===============================
print("Starting Cameras...")

picamL = Picamera2(camera_num=0)
picamL.configure(picamL.create_video_configuration(
    main={"size":image_size,"format":"BGR888"}))
picamL.start()

picamR = Picamera2(camera_num=1)
picamR.configure(picamR.create_video_configuration(
    main={"size":image_size,"format":"BGR888"}))
picamR.start()

# ===============================
# 5. CREATE STEREO MATCHER
# ===============================
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=5,
    P1=8*3*5**2,
    P2=32*3*5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# ===============================
# 6. FPS VARIABLES
# ===============================
prev_time = time.time()
fps = 0

print("Running... Press q to quit")

# ===============================
# 7. MAIN LOOP
# ===============================
try:

    while True:

        # FPS calculation
        current_time = time.time()
        instant_fps = 1/(current_time-prev_time)
        prev_time = current_time
        fps = 0.9*fps + 0.1*instant_fps

        # Capture
        frameL = picamL.capture_array()
        frameR = picamR.capture_array()

        if frameL is None or frameR is None:
            continue

        # Rectify
        left_rect = cv2.remap(frameL,mapLx,mapLy,cv2.INTER_LINEAR)
        right_rect = cv2.remap(frameR,mapRx,mapRy,cv2.INTER_LINEAR)

        # Disparity
        grayL = cv2.cvtColor(left_rect,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_rect,cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(grayL,grayR).astype(np.float32)/16.0
        disparity = cv2.medianBlur(disparity,5)

        # 3D reconstruction
        points_3D = cv2.reprojectImageTo3D(disparity,Q)

        # ===============================
        # YOLO DETECTION (LEFT CAMERA)
        # ===============================
        results = model.predict(
            source=left_rect,
            imgsz=320,
            conf=0.5,
            iou=0.6,
            verbose=False
        )

        display_frame = left_rect.copy()

        for box in results[0].boxes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            if 0<=cx<image_size[0] and 0<=cy<image_size[1]:

                roi = points_3D[
                    max(0,cy-5):min(cy+5,image_size[1]),
                    max(0,cx-5):min(cx+5,image_size[0]),
                    2
                ]

                valid = roi[(roi>0) & (roi<10000)]

                if len(valid)>0:

                    depth_m = np.median(valid)
                    cls = int(box.cls[0])

                    label = f"{model.names[cls]} {depth_m:.2f}m"

                    cv2.rectangle(display_frame,(x1,y1),(x2,y2),(0,255,0),2)

                    cv2.putText(display_frame,
                                label,
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0,255,0),
                                2)

        # Draw FPS
        cv2.putText(display_frame,
                    f"FPS: {fps:.2f}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2)

        cv2.imshow("SamDrishti Stereo - Pi5",display_frame)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

# ===============================
# 8. CLEANUP
# ===============================
finally:
    print("Closing cameras...")
    picamL.stop()
    picamR.stop()
    picamL.close()
    picamR.close()
    cv2.destroyAllWindows()