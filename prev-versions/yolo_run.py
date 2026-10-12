import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

# ==============================
# LOAD YOLO NCNN MODEL
# ==============================
model = YOLO("./yolov8n_ncnn_model", task="detect")

# ==============================
# START LEFT CAMERA
# ==============================
picam2 = Picamera2(0)

config = picam2.create_video_configuration(
    main={
        "size": (640, 480),
        "format": "BGR888"
    }
)

picam2.configure(config)
picam2.start()

print("Camera started. Press Q to quit")

# ==============================
# WARMUP (important for FPS)
# ==============================
dummy = cv2.imread("/usr/share/icons/gnome/256x256/status/dialog-information.png")
model.predict(dummy, imgsz=320, verbose=False)

prev_time = time.time()

# ==============================
# MAIN LOOP
# ==============================
try:

    while True:

        frame = picam2.capture_array()

        # Run YOLO detection
        results = model.predict(
            source=frame,
            imgsz=320,
            conf=0.5,
            iou=0.6,
            classes=[0,1,2,3,13,15,16,56],  # person, bicycle, car, motorcycle, bench, cat, dog, chair
            verbose=False
        )

        annotated = results[0].plot()

        # ==============================
        # FPS
        # ==============================
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            annotated,
            f"FPS: {int(fps)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("YOLO Detection (Left Camera)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()