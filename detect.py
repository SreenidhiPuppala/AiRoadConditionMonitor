from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train3/weights/best.pt")

# Path to your test image
img_path = "datasets/RDD_SPLIT/test/images/China_Drone_000277.jpg"

# Run prediction with lower confidence threshold and IoU for NMS, save output image automatically
results = model.predict(source=img_path, conf=0.01, iou=0.03, save=True)

# Optionally show the first result (predicted image with boxes)
results[0].show()

# Optionally save the result with a custom filename (if not relying on save=True's default path)
results[0].save(filename="outputs/result.jpg")




'''
from ultralytics import YOLO
import os

def run_inference(model_path, image_path, conf_threshold=0.1):
    # Load your trained YOLO model
    model = YOLO(model_path)

    # Run prediction on the image with the specified confidence threshold
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,        # Save the image with predictions drawn
        save_txt=False,   # Optionally save labels in txt format (set True if you want)
        save_conf=True    # Save confidence scores on labels
    )

    # The output image is saved in runs/detect/predict by default
    output_dir = os.path.join('runs', 'detect', 'predict')
    saved_images = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png','.webp'))]

    if saved_images:
        print(f"Inference complete! Output image saved to: {os.path.join(output_dir, saved_images[-1])}")
    else:
        print("No output image found in runs/detect/predict/")

if __name__ == "__main__":
    model_weights = 'runs/detect/train/weights/best.pt'  # Path to your trained model
    test_image = 'datasets/RDD_SPLIT/test/images/United_States_001572.jpg'           # Replace with your test image path

    run_inference(model_weights, test_image, conf_threshold=0.1)
'''