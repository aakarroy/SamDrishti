from picamera2 import Picamera2
import cv2
import time
import os
import numpy as np

# ---------------- SETTINGS ----------------
SAVE_DIR = "stereo_captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- CAMERA INIT ---------------
cam_left = Picamera2(0)
cam_right = Picamera2(1)

cam_left.configure(cam_left.create_preview_configuration())
cam_right.configure(cam_right.create_preview_configuration())

cam_left.start()
cam_right.start()

time.sleep(2)

print("Click on the preview window")
print("Q = Save | ESC = Exit")

img_id = 1

# ---------------- MAIN LOOP ----------------
while True:

    # Capture frames
    left_frame = cam_left.capture_array()
    right_frame = cam_right.capture_array()

    # Fix color (RGB → BGR)
    left_frame = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
    right_frame = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)

    # Resize if sizes don't match (safety)
    if left_frame.shape != right_frame.shape:
        right_frame = cv2.resize(
            right_frame,
            (left_frame.shape[1], left_frame.shape[0])
        )

    # ---- CONNECT SIDE BY SIDE ----
    stereo_view = np.hstack((left_frame, right_frame))


    # Show single window
    cv2.imshow("Stereo Camera (Left | Right)", stereo_view)

    key = cv2.waitKey(1) & 0xFF

    # Save on Q
    if key == ord('q'):
        cv2.imwrite(f"{SAVE_DIR}/L_{img_id:04d}.jpg", left_frame)
        cv2.imwrite(f"{SAVE_DIR}/R_{img_id:04d}.jpg", right_frame)
        print(f"Saved Stereo Pair {img_id}")
        img_id += 1

    # Exit on ESC
    elif key == 27:
        break

# ---------------- CLEANUP ----------------
cam_left.stop()
cam_right.stop()
cv2.destroyAllWindows()

