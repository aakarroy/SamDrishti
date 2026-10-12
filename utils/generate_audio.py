import os
import subprocess

PIPER_EXEC = "/home/samdrishti/codes/piper_app/piper"
PIPER_MODEL = "/home/samdrishti/codes/en_US-lessac-medium.onnx"
OUTPUT_DIR = "/home/samdrishti/codes/audio"

# Create the directory if you haven't already
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dictionary of the exact filenames required by your code and what they should say
ALERTS = {
    # YOLO Classes
    "animal_alert.wav": "Animal detected,",
    "person_alert.wav": "Person detected,",
    "car_alert.wav": "Car detected,",
    "vehicle_alert.wav": "Vehicle detected,",
    "bike_alert.wav": "Bike detected,",
    
    # ESP32 Hardware Overrides
    "fast_alert.wav": "Fast moving object!",
    "pothole_alert.wav": "Pothole detected,",
    "knee_alert.wav": "Obstacle,"
}

print(f"Generating {len(ALERTS)} WAV files for the SamDrishti Pipeline...")

for filename, text in ALERTS.items():
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # The command to pipe text into Piper and save as a WAV file
    cmd = f'echo "{text}" | {PIPER_EXEC} -m {PIPER_MODEL} -f {output_path}'
    
    # Execute the command
    subprocess.run(cmd, shell=True)
    print(f"✅ Created: {filename} -> '{text}'")

print("\n🎉 All required audio files are ready!")