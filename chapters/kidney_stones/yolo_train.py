from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model with your custom dataset, specifying the correct path to your `data.yaml` file
results = model.train(data="data/data.yaml", epochs=100, imgsz=640)