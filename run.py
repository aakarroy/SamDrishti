import multiprocessing as mp
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import time
import subprocess
import os
from collections import deque  # ⭐ NEW: C-optimized rolling window

# ==========================================
# 1. CONFIGURATION
# ==========================================
CALIB_FILE = '/home/samdrishti/codes/stereo_calibration_7x7_filtered_final.xml'
MODEL_PATH = '/home/samdrishti/codes/yolo26n_ncnn_model'

CONF_THRESHOLD = 0.7        
BASELINE = 6.23             
YOLO_SKIP_FRAMES = 2        
DISP_SKIP_FRAMES = 3        
SCALE = 0.5                 
MAX_DETECTIONS = 3          

PIPER_EXEC = "/home/samdrishti/codes/piper_app/piper"
PIPER_MODEL = "/home/samdrishti/codes/en_US-lessac-medium.onnx"
AUDIO_COOLDOWN = 3.0 

# Helper to clear queues and prevent lag buildup
def put_latest(q, item):
    while not q.empty():
        try: q.get_nowait()
        except: pass
    q.put(item)

# ==========================================
# 2. AUDIO PROCESS (Dedicated Core)
# ==========================================
def audio_worker(audio_q):
    """Runs in its own OS process, handling heavy TTS generation."""
    print("[Audio] Process started. Waiting for commands...")
    while True:
        text = audio_q.get() 
        if text is None: break
        
        print(f"\n🎤 AI is saying: {text}") 
        stream_cmd = f'echo "{text}" | {PIPER_EXEC} -m {PIPER_MODEL} -f - | paplay'
        subprocess.run(stream_cmd, shell=True)

# ==========================================
# 3. VISION PROCESS (The Heavy Lifter)
# ==========================================
def vision_worker(raw_frame_q, display_q, audio_q):
    """
    Runs on its own cores. Handles Rectification, YOLO, and StereoBM.
    Models are initialized HERE, not in main, to avoid cross-process pointer corruption.
    """
    print("[Vision] Loading Calibration...")
    cv_file = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
    K1, D1 = cv_file.getNode("camMatrixL").mat(), cv_file.getNode("distL").mat()
    K2, D2 = cv_file.getNode("camMatrixR").mat(), cv_file.getNode("distR").mat()
    R, T = cv_file.getNode("R").mat(), cv_file.getNode("T").mat()
    cv_file.release()

    img_size = (640, 480)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)
    f_pixel = P1[0, 0]

    print("[Vision] Loading YOLO NCNN Model...")
    model = YOLO(MODEL_PATH)
    
    print("[Vision] Initializing StereoBM...")
    # Optimized StereoBM config to save CPU
    stereo = cv2.StereoBM_create(numDisparities=48, blockSize=11)
    stereo.setTextureThreshold(15)   
    stereo.setUniquenessRatio(10)    
    stereo.setSpeckleWindowSize(100) 
    stereo.setSpeckleRange(32)

    def get_depth(disparity_map, x, y, r=5):
        h, w = disparity_map.shape
        x1, x2 = max(0, x-r), min(w, x+r)
        y1, y2 = max(0, y-r), min(h, y+r)
        region = disparity_map[y1:y2, x1:x2]
        vals = region[region > 0]
        if len(vals) == 0: return -1
        disp = np.median(vals) / 16.0
        if disp <= 0: return -1
        return ((f_pixel * SCALE) * BASELINE) / disp

    tick = 0  
    cached_results = []
    cached_disparity = None
    last_speech_time = 0
    
    # ⭐ NEW: Pipeline Constants for smoothing
    MAX_VALID_DEPTH = 2000       # cm
    SMOOTHING_FRAMES = 3         # Rolling average window size
    depth_history = {}           # Dictionary to hold the short-term memory deques
    
    print("[Vision] Pipeline Ready.")

    while True:
        raw_frame = raw_frame_q.get()
        if raw_frame is None: break
        now = time.time()

        # Split and Remap 
        left_raw, right_raw = raw_frame[:, :640], raw_frame[:, 640:]
        left_rect = cv2.remap(left_raw, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, right_map1, right_map2, cv2.INTER_LINEAR)

        # 1. Compute Disparity 
        if tick % DISP_SKIP_FRAMES == 0 or cached_disparity is None: 
            grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
            smallL = cv2.resize(grayL, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            smallR = cv2.resize(grayR, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            cached_disparity = stereo.compute(smallL, smallR)

        # 2. Compute YOLO
        if tick % YOLO_SKIP_FRAMES == 0: 
            rgb = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)
            cached_results = model(rgb, conf=CONF_THRESHOLD, imgsz=416, max_det=MAX_DETECTIONS, verbose=False)

        display = left_rect.copy()
        closest_dist = 9999
        closest_box = None

        # 3. Process Detections
        if cached_results and cached_disparity is not None:
            for r in cached_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (x2 - x1) * (y2 - y1) < 2500: continue

                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    sx, sy = int(cx*SCALE), int(cy*SCALE)
                    raw_d = get_depth(cached_disparity, sx, sy)
                    
                    # ⭐ NEW: OUTLIER REJECTION
                    if raw_d <= 0 or raw_d > MAX_VALID_DEPTH:
                        continue
                    
                    if raw_d < closest_dist:
                        closest_dist = raw_d
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        closest_box = (x1, y1, x2, y2, raw_d, label)

        # 4. Draw & Speak (With Temporal Smoothing)
        if closest_box:
            x1, y1, x2, y2, raw_d, label = closest_box
            
            # ⭐ NEW: Initialize deque if label is seen for the first time
            if label not in depth_history:
                depth_history[label] = deque(maxlen=SMOOTHING_FRAMES)
                
            # ⭐ NEW: Add new raw distance to history and calculate average
            depth_history[label].append(raw_d)
            avg_d = sum(depth_history[label]) / len(depth_history[label])

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{label} | {avg_d:.1f} cm", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if now - last_speech_time > AUDIO_COOLDOWN:
                text_to_speak = f"{label} detected {int(avg_d)} centimeters away."
                put_latest(audio_q, text_to_speak) 
                last_speech_time = now 

        put_latest(display_q, display)
        tick += 1

# ==========================================
# 4. MAIN OS PROCESS (I/O & Display Loop)
# ==========================================
if __name__ == '__main__':
    # CRITICAL: Remember to run this from the Linux Terminal (python3 script.py), 
    # NOT inside an IDE like Thonny, to avoid the spawn/attribute error.
    mp.set_start_method('spawn')
    
    raw_frame_q = mp.Queue(maxsize=2)
    display_q = mp.Queue(maxsize=2)
    audio_q = mp.Queue(maxsize=2)

    audio_proc = mp.Process(target=audio_worker, args=(audio_q,), daemon=True)
    vision_proc = mp.Process(target=vision_worker, args=(raw_frame_q, display_q, audio_q), daemon=True)
    
    audio_proc.start()
    vision_proc.start()

    print("[Main] Starting Picamera2...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 480), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    picam2.set_controls({"AeEnable": False}) 

    print("🚀 SamDrishti MP Architecture Running... Press 'q' to quit")
    
    frame_count = 0
    prev_time = time.time()
    fps = 0

    try:
        while True:
            raw_frame = picam2.capture_array()
            if raw_frame is not None:
                put_latest(raw_frame_q, raw_frame)

            if not display_q.empty():
                display_frame = display_q.get()
                
                now = time.time()
                frame_count += 1
                if now - prev_time > 1:
                    fps = frame_count / (now - prev_time)
                    frame_count = 0  
                    prev_time = now

                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("SamDrishti Vision", display_frame)
            
            if cv2.waitKey(1) == ord('q'): 
                break

    finally:
        print("Shutting down processes...")
        audio_q.put(None)
        raw_frame_q.put(None)
        
        audio_proc.join(timeout=2)
        vision_proc.join(timeout=2)
        
        if audio_proc.is_alive(): audio_proc.terminate()
        if vision_proc.is_alive(): vision_proc.terminate()
        
        picam2.stop()
        cv2.destroyAllWindows()
        print("✅ Shutdown complete")