from ultralytics import YOLO

model = YOLO("hope-to-god.pt")
model.predict(source='Inside a Wildfire - Dramatic Drone Footage! Ep. 193a..mp4', imgsz=640, conf=0.25, save=True)