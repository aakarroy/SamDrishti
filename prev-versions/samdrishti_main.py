import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import multiprocessing as mp
import multiprocessing.synchronize
from multiprocessing import shared_memory
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import subprocess
import json
import serial
import queue
from collections import deque
import sys
import base64
from typing import Optional

# ==========================================
# 1. CONFIGURATION
# ==========================================
CALIB_FILE = '/home/samdrishti/codes/stereo_calibration_7x7_filtered_final.xml'
MODEL_PATH = '/home/samdrishti/codes/best_ncnn_model'
YOLO_SCRIPT = '/home/samdrishti/codes/yolo_inference.py'
PYTHON_EXEC = sys.executable
ESP32_PORT = '/dev/ttyUSB0'
ESP32_BAUD = 115200

CONF_THRESHOLD = 0.7
BASELINE = 6.23
YOLO_SKIP_FRAMES = 2
DISP_SKIP_FRAMES = 3
SCALE = 0.5
MAX_DETECTIONS = 3

PIPER_EXEC = "/home/samdrishti/codes/piper_app/piper"
PIPER_MODEL = "/home/samdrishti/codes/en_US-lessac-medium.onnx"
AUDIO_COOLDOWN = 3.0
AUDIO_DIR = "/home/samdrishti/codes/audio"
OVERRIDE_COOLDOWN = 1.5

RAW_SHAPE = (480, 1280, 3)
DISP_SHAPE = (480, 640, 3)

# ==========================================
# 2. ESP32 HARDWARE WORKER
# ==========================================
def esp32_worker(in_esp_q: "mp.Queue[dict]") -> None:
    print(f"[ESP32] Connecting to {ESP32_PORT} at {ESP32_BAUD} baud...")
    try:
        ser = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
        print("[ESP32] Hardware Bridge Established.")
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                try:
                    data = json.loads(line)
                    msg_type = data.get("type", "")
                    if msg_type == "status":
                        print(f"🔧 [ESP32 Status] {data.get('msg')}")
                    elif msg_type in ["fast", "pothole", "obstacle"]:
                        try:
                            in_esp_q.put_nowait(data)
                        except queue.Full:
                            pass
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"⚠️ [ESP32] ERROR: {e}. Running without hardware sensors.")

# ==========================================
# 3. AUDIO PROCESS
# ==========================================
def audio_worker(in_audio_q: "mp.Queue[Optional[dict]]") -> None:
    print("[Audio] Process started. Waiting for commands...")
    tmp_wav = "/tmp/distance.wav"
    while True:
        payload = in_audio_q.get()
        if payload is None:
            break
        wav_file = payload.get("wav", None)
        distance_text = payload.get("text", None)
        print(f"\n🎤 Announcing: {wav_file} -> {distance_text}")
        if wav_file and os.path.exists(wav_file):
            subprocess.Popen(
                ["paplay", wav_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if distance_text:
            gen_cmd = f'echo "{distance_text}" | {PIPER_EXEC} -m {PIPER_MODEL} -f {tmp_wav}'
            subprocess.run(gen_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(tmp_wav):
                subprocess.Popen(
                    ["paplay", tmp_wav],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

# ==========================================
# 4. VISION PROCESS
# ==========================================
def vision_worker(
    shm_raw_name: str,
    shm_disp_name: str,
    in_signal_q: "mp.Queue[Optional[bool]]",
    in_result_q: "mp.Queue[bool]",
    in_audio_q: "mp.Queue[Optional[dict]]",
    in_esp_q: "mp.Queue[dict]",
    in_frame_lock: multiprocessing.synchronize.Lock,
) -> None:
    cv2.setNumThreads(1)

    print("[Vision] Loading Calibration...")
    cv_file = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
    K1 = cv_file.getNode("camMatrixL").mat()
    D1 = cv_file.getNode("distL").mat()
    K2 = cv_file.getNode("camMatrixR").mat()
    D2 = cv_file.getNode("distR").mat()
    R  = cv_file.getNode("R").mat()
    T  = cv_file.getNode("T").mat()
    cv_file.release()

    img_size = (640, 480)
    R1, R2, P1, P2, _Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T)
    left_map1, left_map2   = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)
    f_pixel = P1[0, 0]

    print("[Vision] Initializing StereoBM...")
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)  # type: ignore[attr-defined]
    stereo.setTextureThreshold(15)
    stereo.setUniquenessRatio(10)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(32)

    def get_depth(disparity_map: np.ndarray, x: int, y: int, r: int = 5) -> float:
        h, w = disparity_map.shape
        x1, x2 = max(0, x - r), min(w - 1, x + r)
        y1, y2 = max(0, y - r), min(h - 1, y + r)
        region = disparity_map[y1:y2, x1:x2]
        vals = region[region > 0]
        if len(vals) == 0:
            return -1.0
        disp = np.median(vals) / 16.0
        if disp <= 0:
            return -1.0
        return float(((f_pixel * SCALE) * BASELINE) / disp)

    def start_yolo() -> subprocess.Popen:
        proc = subprocess.Popen(
            [PYTHON_EXEC, YOLO_SCRIPT, MODEL_PATH,
             str(CONF_THRESHOLD), "320", str(MAX_DETECTIONS)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open("/tmp/yolo_stderr.log", "w"),  # log crashes for diagnosis
            text=True,
            bufsize=1,
        )
        if proc.stdout:
            for line in proc.stdout:
                if line.strip() == "READY":
                    break
                print(f"[YOLO startup] {line.strip()}", flush=True)
        return proc

    print("[Vision] Starting persistent YOLO subprocess...")
    yolo_proc = start_yolo()
    print("[Vision] YOLO subprocess ready.")

    tick: int = 0
    cached_detections: list = []
    cached_disparity: Optional[np.ndarray] = None
    last_speech_time: float = 0.0
    last_urgent_time: float = 0.0

    MAX_VALID_DEPTH = 2000
    SMOOTHING_FRAMES = 3
    depth_history: dict = {}

    shm_raw_mem  = shared_memory.SharedMemory(name=shm_raw_name)
    shm_disp_mem = shared_memory.SharedMemory(name=shm_disp_name)
    in_raw_frame:      np.ndarray = np.ndarray(RAW_SHAPE,  dtype=np.uint8, buffer=shm_raw_mem.buf)
    display_frame_out: np.ndarray = np.ndarray(DISP_SHAPE, dtype=np.uint8, buffer=shm_disp_mem.buf)

    print("[Vision] Pipeline Ready.")

    while True:
        token = in_signal_q.get()
        if token is None:
            break
        frame_time: float = time.time()

        with in_frame_lock:
            left_raw  = in_raw_frame[:, :640].copy()
            right_raw = in_raw_frame[:, 640:].copy()

        left_rect  = cv2.remap(left_raw,  left_map1,  left_map2,  cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, right_map1, right_map2, cv2.INTER_LINEAR)

        # StereoBM
        if tick % DISP_SKIP_FRAMES == 0 or cached_disparity is None:
            grayL  = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)
            grayR  = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
            smallL = cv2.resize(grayL, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            smallR = cv2.resize(grayR, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            result = stereo.compute(smallL, smallR)
            if result is not None and result.shape[0] > 0:
                cached_disparity = result

        # YOLO inference
        if tick % YOLO_SKIP_FRAMES == 0:
            # Check if subprocess died and print why before restarting
            if yolo_proc.poll() is not None:
                try:
                    with open("/tmp/yolo_stderr.log", "r") as _f:
                        _err = _f.read().strip()
                        if _err:
                            print(f"[Vision] YOLO crash reason:\n{_err}", flush=True)
                except Exception:
                    pass
                print("[Vision] YOLO subprocess died, restarting...", flush=True)
                yolo_proc = start_yolo()
                print("[Vision] YOLO restarted.", flush=True)

            try:
                _, buf = cv2.imencode('.jpg', left_rect, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buf).decode('ascii')
                if yolo_proc.stdin:
                    yolo_proc.stdin.write(b64 + '\n')
                    yolo_proc.stdin.flush()
                if yolo_proc.stdout:
                    result_line = yolo_proc.stdout.readline().strip()
                    if result_line and result_line.startswith('['):
                        cached_detections = json.loads(result_line)
                    # empty/bad read: keep last cached_detections, don't reset
            except Exception as e:
                print(f"[Vision] YOLO pipe error: {e}", flush=True)
                # keep cached_detections — don't clear on transient error

        display = left_rect.copy()
        closest_dist: float = 9999.0
        closest_box = None

        if cached_detections and cached_disparity is not None:
            for det in cached_detections:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                if (x2 - x1) * (y2 - y1) < 2500:
                    continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                sx, sy = int(cx * SCALE), int(cy * SCALE)
                raw_d = get_depth(cached_disparity, sx, sy)
                if raw_d <= 0 or raw_d > MAX_VALID_DEPTH:
                    continue
                if raw_d < closest_dist:
                    closest_dist = raw_d
                    detected_class = det["cls_name"]
                    closest_box = (x1, y1, x2, y2, raw_d, detected_class)

        if closest_box:
            x1, y1, x2, y2, raw_d, detected_class = closest_box
            if detected_class not in depth_history:
                depth_history[detected_class] = deque(maxlen=SMOOTHING_FRAMES)
            depth_history[detected_class].append(raw_d)
            avg_d = sum(depth_history[detected_class]) / len(depth_history[detected_class])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{detected_class} | {avg_d:.1f} cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            closest_box = (x1, y1, x2, y2, avg_d, detected_class)

        # PRIORITY ENGINE
        alert_payload = None
        priority_level = 99
        esp_data = None
        if not in_esp_q.empty():
            try:
                esp_data = in_esp_q.get_nowait()
            except queue.Empty:
                pass

        if esp_data and esp_data.get("type") == "fast":
            alert_payload = {
                "wav": f"{AUDIO_DIR}/fast_alert.wav",
                "text": f"Incoming at {esp_data.get('pos', 'FRONT')}",
            }
            priority_level = 1
        elif esp_data and esp_data.get("type") == "pothole":
            alert_payload = {
                "wav": f"{AUDIO_DIR}/pothole_alert.wav",
                "text": str(esp_data.get("dist", "")),
            }
            priority_level = 2
        elif closest_box:
            alert_payload = {
                "wav": f"{AUDIO_DIR}/{closest_box[5]}_alert.wav",
                "text": f"{int(closest_box[4])} centimeters away",
            }
            priority_level = 3
        elif esp_data and esp_data.get("type") == "obstacle":
            alert_payload = {
                "wav": f"{AUDIO_DIR}/knee_alert.wav",
                "text": f"Obstacle at {esp_data.get('pos', 'front')}, {esp_data.get('dist', 'unknown')} cm.",
            }
            priority_level = 4

        if alert_payload:
            if priority_level <= 2 and (frame_time - last_urgent_time > OVERRIDE_COOLDOWN):
                try:
                    in_audio_q.put_nowait(alert_payload)
                    last_urgent_time = frame_time
                    last_speech_time = frame_time
                except queue.Full:
                    pass
            elif priority_level >= 3 and (frame_time - last_speech_time > AUDIO_COOLDOWN):
                try:
                    in_audio_q.put_nowait(alert_payload)
                    last_speech_time = frame_time
                except queue.Full:
                    pass

        np.copyto(display_frame_out, display)
        try:
            in_result_q.put_nowait(True)
        except queue.Full:
            pass

        tick += 1

    # Cleanup
    if yolo_proc.stdin:
        yolo_proc.stdin.close()
    yolo_proc.terminate()
    del in_raw_frame
    del display_frame_out
    shm_raw_mem.close()
    shm_disp_mem.close()
# ==========================================
# 5. MAIN OS PROCESS
# ==========================================
if __name__ == '__main__':
    import subprocess as _sp

    # Only kill known camera daemons — NOT 'samdrishti' or 'python3'
    # (lsof showed /dev/media is free; we just need to guard against stuck rpicam/libcamera)
    print("[Main] Releasing camera device...")
    for _proc in ["rpicam", "libcamera", "picamera"]:
        _sp.run(["sudo", "pkill", "-9", "-f", _proc], stderr=_sp.DEVNULL)
    time.sleep(1.0)

    mp.set_start_method('spawn')

    raw_size  = int(np.prod(RAW_SHAPE))
    disp_size = int(np.prod(DISP_SHAPE))

    SHM_RAW_NAME  = "samdrishti_raw"
    SHM_DISP_NAME = "samdrishti_disp"

    def safe_init_shm(name: str, size: int) -> shared_memory.SharedMemory:
        try:
            stuck_shm = shared_memory.SharedMemory(name=name)
            stuck_shm.unlink()
            print(f"🧹 Cleaned up stuck shared memory: {name}")
        except FileNotFoundError:
            pass
        return shared_memory.SharedMemory(name=name, create=True, size=size)

    shm_raw  = safe_init_shm(SHM_RAW_NAME,  raw_size)
    shm_disp = safe_init_shm(SHM_DISP_NAME, disp_size)

    main_raw_frame:  np.ndarray = np.ndarray(RAW_SHAPE,  dtype=np.uint8, buffer=shm_raw.buf)
    main_disp_frame: np.ndarray = np.ndarray(DISP_SHAPE, dtype=np.uint8, buffer=shm_disp.buf)

    signal_q: "mp.Queue[Optional[bool]]" = mp.Queue(maxsize=1)
    result_q: "mp.Queue[bool]"           = mp.Queue(maxsize=1)
    audio_q:  "mp.Queue[Optional[dict]]" = mp.Queue(maxsize=2)
    esp_q:    "mp.Queue[dict]"           = mp.Queue(maxsize=2)
    frame_lock: multiprocessing.synchronize.Lock = mp.Lock()

    # ✅ Initialize camera FIRST, before spawning any child processes.
    # Child processes must never touch the camera — they only read shared memory.
    print("[Main] Starting Picamera2...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 480), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    picam2.set_controls({"AeEnable": False})
    print("[Main] Camera ready.")

    # ✅ Spawn children only after camera is locked by main process
    audio_proc  = mp.Process(target=audio_worker, args=(audio_q,), daemon=True)
    vision_proc = mp.Process(
        target=vision_worker,
        args=(shm_raw.name, shm_disp.name, signal_q, result_q, audio_q, esp_q, frame_lock),
        daemon=True,
    )
    esp_proc = mp.Process(target=esp32_worker, args=(esp_q,), daemon=True)

    audio_proc.start()
    vision_proc.start()
    esp_proc.start()

    print("🚀 SamDrishti MP Architecture Running... Press 'q' to quit")

    frame_count: int   = 0
    prev_time:   float = time.time()
    fps:         float = 0.0
    local_display: np.ndarray = np.zeros(DISP_SHAPE, dtype=np.uint8)

    try:
        while True:
            captured = picam2.capture_array()
            if captured is not None:
                if signal_q.empty():
                    with frame_lock:
                        np.copyto(main_raw_frame, captured)
                    try:
                        signal_q.put_nowait(True)
                    except queue.Full:
                        pass
                else:
                    time.sleep(0.005)

            if not result_q.empty():
                try:
                    result_q.get_nowait()
                except queue.Empty:
                    pass
                np.copyto(local_display, main_disp_frame)
                loop_now: float = time.time()
                frame_count += 1
                if loop_now - prev_time > 1:
                    fps = frame_count / (loop_now - prev_time)
                    frame_count = 0
                    prev_time = loop_now

            cv2.putText(local_display, f"AI FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("SamDrishti Vision", local_display)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        print("Shutting down processes...")
        try:
            audio_q.put_nowait(None)
        except queue.Full:
            pass
        try:
            signal_q.put_nowait(None)
        except queue.Full:
            pass
        time.sleep(0.5)
        if audio_proc.is_alive():  audio_proc.terminate()
        if vision_proc.is_alive(): vision_proc.terminate()
        if esp_proc.is_alive():    esp_proc.terminate()
        picam2.stop()
        cv2.destroyAllWindows()
        del main_raw_frame
        del main_disp_frame
        shm_raw.close();  shm_raw.unlink()
        shm_disp.close(); shm_disp.unlink()
        print("✅ Shutdown complete")