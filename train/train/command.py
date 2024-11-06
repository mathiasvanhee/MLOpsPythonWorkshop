import argparse
from pathlib import Path

import mlflow
from ultralytics import YOLO

parser = argparse.ArgumentParser("train")
parser.add_argument("--dataset", type=str)
parser.add_argument("--model_output", type=str)

# Get arguments from parser
args = parser.parse_args()
dataset = args.dataset
model_output = args.model_output

batch_size = 16
epochs = 30
img_size = 640
params = {
    "batch_size": batch_size,
    "epochs": epochs,
    "img_size": img_size,
}
mlflow.log_params(params)

model = YOLO('yolov8n.pt')  # V8 nano
model.train(data=Path(dataset) / "data.yaml", epochs=epochs, batch=batch_size, imgsz=img_size)
model.save(model_output)

# mlflow.log_image(Image.open(result.summary_image_path), "figure.png")
# mlflow.log_metric("evaluate_accuracy_percentage", result.evaluate_accuracy_percentage)
