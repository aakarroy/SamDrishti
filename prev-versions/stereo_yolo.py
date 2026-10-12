import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import time
import threading
import queue

# ==========================================
# 1. CONFIGURATION
# ==========================================
CALIB_FILE = '/home/samdrishti/codes/stereo_calibration_7x7_filtered_final.xml'
MODEL_PATH = '/home/samdrishti/codes/yolo26n_ncnn_model'

CONF_THRESHOLD = 0.7        # High confidence for assistive reliability
BASELINE = 6.23             # cm between camera lenses

YOLO_SKIP_FRAMES = 2        # Run AI every 3rd frame
DISP_SKIP_FRAMES = 3        # Run Depth every 4th frame
SCALE = 0.5                 # Shrink disparity input by 50% for speed

MAX_DETECTIONS = 3          # Hard limit to save CPU

# ==========================================
# 2. LOAD CALIBRATION
# ==========================================
print("Loading calibration data...")
cv_file = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)

def load_matrix(name):
    return cv_file.getNode(name).mat()

K1 = load_matrix("camMatrixL")
D1 = load_matrix("distL")
K2 = load_matrix("camMatrixR")
D2 = load_matrix("distR")
R = load_matrix("R")
T = load_matrix("T")
cv_file.release()

img_size = (640, 480)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)

f_pixel = P1[0, 0]

# ==========================================
# 3. LOAD YOLO & STEREO
# ==========================================
print("Loading YOLO NCNN Model...")
model = YOLO(MODEL_PATH)

print("Initializing StereoBM...")
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)

# ==========================================
# 4. DEPTH FUNCTION (WITH MATH FIX)
# ==========================================
def get_depth(disparity_map, x, y, r=5):
    h, w = disparity_map.shape
    x1, x2 = max(0, x-r), min(w, x+r)
    y1, y2 = max(0, y-r), min(h, y+r)
    
    region = disparity_map[y1:y2, x1:x2]
    vals = region[region > 0]
    
    if len(vals) == 0:
        return -1
        
    disp = np.median(vals) / 16.0
    if disp <= 0:
        return -1
        
    # ⭐ FIXED: Multiply f_pixel by SCALE because the disparity map was shrunk!
    return ((f_pixel * SCALE) * BASELINE) / disp

# ==========================================
# 5. CAMERA THREAD (Producer)
# ==========================================
print("Starting Picamera2 & Threading...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 480), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)
picam2.set_controls({"AeEnable": False}) # Lock exposure for stable FPS

# Thread-safe queue to hold the freshest frame
frame_queue = queue.Queue(maxsize=1)

def camera_worker():
    while True:
        frame = picam2.capture_array()
        if frame is None:
            continue
            
        left_raw = frame[:, :640]
        right_raw = frame[:, 640:]
        
        # Rectify instantly on the background thread
        left_rect = cv2.remap(left_raw, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, right_map1, right_map2, cv2.INTER_LINEAR)
        
        # Keep only the newest frame in the bucket
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put((left_rect, right_rect))

# Start the background camera thread
cam_thread = threading.Thread(target=camera_worker, daemon=True)
cam_thread.start()
print("✅ Multi-threaded Camera ready")

# ==========================================
# 6. MAIN AI LOOP (Consumer)
# ==========================================
frame_count = 0
prev_time = time.time()
fps = 0

cached_results = []
cached_disparity = None

print("🚀 Running SamDrishti Vision... Press 'q' to quit")

try:
    while True:
        # Grab the absolute freshest frame from the camera thread
        left_rect, right_rect = frame_queue.get()

        grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        smallL = cv2.resize(grayL, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        smallR = cv2.resize(grayR, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

        # 1. Compute Disparity
        if frame_count % DISP_SKIP_FRAMES == 0 or cached_disparity is None:
            cached_disparity = stereo.compute(smallL, smallR)

        # 2. Compute YOLO
        if frame_count % YOLO_SKIP_FRAMES == 0:
            rgb = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)
            cached_results = model(
                rgb,
                conf=CONF_THRESHOLD,
                imgsz=416,  # Force smaller input size for NCNN speed
                max_det=MAX_DETECTIONS,
                verbose=False
            )

        display = left_rect.copy()
        closest_dist = 9999
        closest_box = None

        # 3. Process Detections & Find Closest
        if cached_results and cached_disparity is not None:
            for r in cached_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Filter tiny background boxes
                    area = (x2 - x1) * (y2 - y1)
                    if area < 2500:
                        continue

                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    sx, sy = int(cx*SCALE), int(cy*SCALE)

                    d = get_depth(cached_disparity, sx, sy)
                    
                    # Track the single closest object
                    if 0 < d < closest_dist:
                        closest_dist = d
                        # Also grab the label to display what the object is
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        closest_box = (x1, y1, x2, y2, d, label)

        # 4. Draw ONLY the closest object
        if closest_box:
            x1, y1, x2, y2, d, label = closest_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{label} | {d:.1f} cm", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(display, ((x1+x2)//2, (y1+y2)//2), 5, (0, 255, 0), -1)

        # 5. FPS Calc & Display
        frame_count += 1
        now = time.time()
        if now - prev_time > 1:
            fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now

        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("SamDrishti Vision", display)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("✅ Shutdown complete")