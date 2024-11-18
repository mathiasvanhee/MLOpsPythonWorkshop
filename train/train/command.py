import argparse
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser("train")
parser.add_argument("--dataset", type=str)
parser.add_argument("--model_output", type=str)

# Get arguments from parser
args = parser.parse_args()
dataset = args.dataset
model_output = args.model_output

batch_size = 16
epochs = 2
img_size = 640
model = YOLO('yolov8n.pt')  # V8 nano
model.train(data=Path(dataset) / "data.yaml", epochs=epochs, batch=batch_size, imgsz=img_size)
model.save(model_output)

