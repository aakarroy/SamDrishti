import os
import multiprocessing as mp
import time
import subprocess
import json
import serial
import queue

# ==========================================
# 1. CONFIGURATION
# ==========================================
ESP32_PORT = '/dev/ttyUSB0'
ESP32_BAUD = 115200

PIPER_EXEC = "/home/samdrishti/codes/piper_app/piper"
PIPER_MODEL = "/home/samdrishti/codes/en_US-lessac-medium.onnx"
PIPER_SPEED = 0.8 

AUDIO_COOLDOWN = 3.0       
OVERRIDE_COOLDOWN = 1.5    

# ==========================================
# 2. ESP32 HARDWARE WORKER
# ==========================================
def esp32_worker(esp_q):
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
                    # Now explicitly catching "clear" as well
                    if msg_type == "status":
                        print(f"🔧 [ESP32 Status] {data.get('msg')}")
                    elif msg_type in ["fast", "pothole", "obstacle", "clear"]:
                        try:
                            esp_q.put_nowait(data)
                        except queue.Full:
                            pass 
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"⚠️ [ESP32] ERROR: {e}. Cannot connect to ESP32.")

# ==========================================
# 3. AUDIO PROCESS
# ==========================================
def audio_worker(audio_q):
    print("[Audio] Process started. Waiting for commands...")
    tmp_wav = "/tmp/distance.wav"
    
    while True:
        payload = audio_q.get()
        if payload is None: break
        
        distance_text = payload.get("text", None)
        if not distance_text:
            continue
            
        print(f"\n🎤 Piper Generating: {distance_text}")
            
        gen_cmd = f'echo "{distance_text}" | {PIPER_EXEC} -m {PIPER_MODEL} --length_scale {PIPER_SPEED} -f {tmp_wav}'
        subprocess.run(gen_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(tmp_wav):
            subprocess.Popen(["paplay", tmp_wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 4. MAIN LOGIC ENGINE
# ==========================================
if __name__ == '__main__':
    mp.set_start_method('spawn')

    # ⭐ FIX 5: Increased queue size to prevent dropping rapid-fire events
    audio_q  = mp.Queue(maxsize=10)
    esp_q    = mp.Queue(maxsize=10)

    audio_proc  = mp.Process(target=audio_worker, args=(audio_q,), daemon=True)
    esp_proc    = mp.Process(target=esp32_worker, args=(esp_q,), daemon=True)

    audio_proc.start()
    esp_proc.start()

    print("🚀 SamDrishti ESP32 Audio Bridge Running... Press Ctrl+C to quit")

    last_speech_time = 0
    last_urgent_time = 0

    try:
        while True:
            esp_data = None
            
            if not esp_q.empty():
                try:
                    esp_data = esp_q.get_nowait()
                except queue.Empty:
                    pass

            if esp_data:
                now = time.time()
                alert_payload = None
                priority_level = 99
                
                msg_type = esp_data.get("type")
                
                # ⭐ FIX 3: Normalize position text to lowercase for consistent TTS
                pos = str(esp_data.get("pos", "front")).lower()
                
                # ⭐ FIX 1: Safely convert ESP32 mm to cm
                raw_dist = esp_data.get("dist", "unknown")
                if str(raw_dist).isdigit():
                    dist_cm = int(raw_dist) // 10
                else:
                    dist_cm = "unknown"

                # Priority Routing & Payload Generation
                msg_type = esp_data.get("type")
                
                if msg_type == "fast":
                    alert_payload = {"text": f"Incoming fast object at {esp_data.get('pos', 'front').lower()}"}
                    priority_level = 1
                    
                elif msg_type == "pothole":
                    # Applied your inline conversion and context fix
                    alert_payload = {"text": f"Pothole ahead, {round(esp_data.get('dist', 0) / 10)} centimeters."}
                    priority_level = 2
                    
                elif msg_type == "obstacle":
                    # Applied your inline conversion fix
                    alert_payload = {"text": f"Obstacle at {esp_data.get('pos', 'front').lower()}, {round(esp_data.get('dist', 0) / 10)} centimeters."}
                    priority_level = 4
                    
                elif msg_type == "clear":
                    # ⭐ FIX 4: Explicitly handle clear state (kept silent for TTS sanity)
                    pass

                # Dispatch to Audio Queue based on Cooldown Rules
                if alert_payload:
                    if priority_level <= 2 and (now - last_urgent_time > OVERRIDE_COOLDOWN):
                        try:
                            audio_q.put_nowait(alert_payload)
                            last_urgent_time = now
                            last_speech_time = now
                        except queue.Full:
                            pass
                            
                    elif priority_level >= 3 and (now - last_speech_time > AUDIO_COOLDOWN):
                        try:
                            audio_q.put_nowait(alert_payload)
                            last_speech_time = now
                        except queue.Full:
                            pass

            time.sleep(0.01) 

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C detected. Shutting down cleanly...")

    finally:
        print("Shutting down processes...")
        
        try: 
            audio_q.put_nowait(None)
        except queue.Full: 
            pass
            
        time.sleep(0.5)
        
        if audio_proc.is_alive():  audio_proc.terminate()
        if esp_proc.is_alive():    esp_proc.terminate()
        
        print("✅ Shutdown complete")