from ultralytics import YOLO

# Load your trained model
model = YOLO("/home/samdrishti/codes/best.pt")   # your YOLOv8 model

# Export to NCNN
model.export(format="ncnn",imgsz=416)