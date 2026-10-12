import sys
import json
import base64
import numpy as np
import cv2
from ultralytics import YOLO

sys.stdout.reconfigure(line_buffering=True)
# Args: model_path conf imgsz max_det  (4 args = sys.argv[1..4])
model_path: str = sys.argv[1]
conf:       float = float(sys.argv[2])
imgsz:      int   = int(sys.argv[3])
max_det:    int   = int(sys.argv[4])

model = YOLO(model_path, task='detect')

# Flush all startup noise first, then print READY on its own line
sys.stdout.write("READY\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        img_bytes = base64.b64decode(line)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(rgb, conf=conf, imgsz=imgsz, max_det=max_det, verbose=False)

        detections: list = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id: int = int(box.cls[0])
                detections.append({
                    "x1":       x1,
                    "y1":       y1,
                    "x2":       x2,
                    "y2":       y2,
                    "cls_id":   cls_id,
                    "cls_name": model.names[cls_id],
                })

        sys.stdout.write(json.dumps(detections) + "\n")
        sys.stdout.flush()

    except Exception as e:
        sys.stdout.write("[]\n")
        sys.stdout.flush()