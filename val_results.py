from ultralytics import YOLO

# Load your best trained model
model = YOLO('runs/detect/train3/weights/best.pt')

# Validate the model and save class-wise metrics to JSON
model.val(save_json=True, data='data.yaml')
