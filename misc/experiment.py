import cv2
import numpy as np
import time
import threading
import queue
from picamera2 import Picamera2
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
CALIB_FILE = '/home/samdrishti/codes/stereo_calibration_2.xml'
MODEL_PATH = '/home/samdrishti/codes/yolov8n_ncnn_model' # Nano model

CONF_THRESHOLD = 0.6        # Confidence threshold
BASELINE = 6.04            # cm between camera lenses (Adjust if needed!)

YOLO_SKIP_FRAMES = 2        # Run AI every 2nd frame
DISP_SKIP_FRAMES = 3        # Run Depth every 3rd frame
SCALE = 0.5                 # Shrink disparity input by 50% for speed

MAX_DETECTIONS = 3          # Hard limit to save CPU
IMAGE_SIZE = (640, 480)

# Force OpenCV to use all CPU cores

# ==========================================
# 2. LOAD CALIBRATION
# ==========================================
print("Loading calibration data...")
cv_file = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)

K1 = cv_file.getNode("camMatrixL").mat()
D1 = cv_file.getNode("distL").mat()
K2 = cv_file.getNode("camMatrixR").mat()
D2 = cv_file.getNode("distR").mat()
R = cv_file.getNode("R").mat()
T = cv_file.getNode("T").mat()
cv_file.release()

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, IMAGE_SIZE, R, T)

# Precompute maps (16-bit integer for speed)
mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, IMAGE_SIZE, cv2.CV_16SC2)
mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, R2, P2, IMAGE_SIZE, cv2.CV_16SC2)

# Extract focal length for manual depth calculation
f_pixel = P1[0, 0]

# ==========================================
# 3. DEPTH FUNCTION 
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
        
    # Multiply f_pixel by SCALE because the disparity map was shrunk!
    return ((f_pixel * SCALE) * BASELINE) / disp

# ==========================================
# 4. FAST DUAL-CAMERA QUEUE THREAD
# ==========================================
class FastCameraThread:
    def __init__(self):
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)
        
        print("Starting Picamera2 modules...")
        self.picamL = Picamera2(camera_num=0)
        self.picamL.configure(self.picamL.create_video_configuration(
            main={"size": IMAGE_SIZE, "format": "BGR888"}))
        self.picamL.start()

        self.picamR = Picamera2(camera_num=1)
        self.picamR.configure(self.picamR.create_video_configuration(
            main={"size": IMAGE_SIZE, "format": "BGR888"}))
        self.picamR.start()

        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        print("✅ Multi-threaded Cameras ready")

    def worker(self):
        while self.running:
            frameL = self.picamL.capture_array()
            frameR = self.picamR.capture_array()
            
            if frameL is None or frameR is None:
                continue

            # Rectify instantly on the background thread
            left_rect = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
            right_rect = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

            # Keep only the newest frame in the bucket
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((left_rect, right_rect))

    def get(self):
        return self.frame_queue.get()

    def stop(self):
        self.running = False
        self.thread.join()
        self.picamL.stop()
        self.picamR.stop()

# ==========================================
# 5. LOAD YOLO & STEREO
# ==========================================
print("Loading YOLO NCNN Model...")
model = YOLO(MODEL_PATH, task="detect")

print("Initializing Fast Stereo...")
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=32,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # Faster algorithm
)

cam_stream = FastCameraThread()
time.sleep(1.0) # Warmup

# ==========================================
# 6. MAIN AI LOOP
# ==========================================
frame_count = 0
prev_time = time.time()
fps = 0

cached_results = []
cached_disparity = None

print("🚀 Running SamDrishti Vision... Press 'q' to quit")

try:
    while True:
        # Grab the absolute freshest frame
        left_rect, right_rect = cam_stream.get()

        # ---------- 1. COMPUTE DISPARITY (Skipping frames) ----------
        if frame_count % DISP_SKIP_FRAMES == 0 or cached_disparity is None:
            grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

            smallL = cv2.resize(grayL, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
            smallR = cv2.resize(grayR, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)

            cached_disparity = stereo.compute(smallL, smallR).astype(np.float32)

        # ---------- 2. COMPUTE YOLO (Skipping frames) ----------
        if frame_count % YOLO_SKIP_FRAMES == 0:
            cached_results = model(
                left_rect, 
                conf=CONF_THRESHOLD, 
                imgsz=320, 
                max_det=MAX_DETECTIONS, 
                verbose=False,
                device=0
            )

        display = left_rect.copy()
        closest_dist = 9999
        closest_box = None

        # ---------- 3. PROCESS & FIND CLOSEST ----------
        if cached_results and cached_disparity is not None:
            for r in cached_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Filter tiny background boxes to save math
                    area = (x2 - x1) * (y2 - y1)
                    if area < 2500:
                        continue

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    sx, sy = int(cx * SCALE), int(cy * SCALE)

                    d = get_depth(cached_disparity, sx, sy)
                    
                    # Track the single closest object
                    if 0 < d < closest_dist:
                        closest_dist = d
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        closest_box = (x1, y1, x2, y2, d, label)

        # ---------- 4. DRAW CLOSEST OBJECT ----------
        if closest_box:
            x1, y1, x2, y2, d, label = closest_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{label} | {d:.1f} cm", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(display, ((x1+x2)//2, (y1+y2)//2), 5, (0, 255, 0), -1)

        # ---------- 5. FPS CALCULATION ----------
        frame_count += 1
        now = time.time()
        if now - prev_time > 1:
            fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now

        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("SamDrishti Vision", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Closing cameras...")
    cam_stream.stop()
    cv2.destroyAllWindows()
    print("✅ Shutdown complete")