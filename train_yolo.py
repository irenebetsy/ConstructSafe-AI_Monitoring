import os
import torch

# Specify paths for your data
data_yaml_path = 'D:/cvprojects/Hard Hat Workers.v2-raw.yolov5pytorch.zip/data.yaml'  # Path to data.yaml
train_images_path = 'D:/cvprojects/Hard Hat Workers.v2-raw.yolov5pytorch.zip/train/images'  # Path to train images
train_labels_path = 'D:/cvprojects/Hard Hat Workers.v2-raw.yolov5pytorch.zip/train/labels'  # Path to train labels

# Change working directory to YOLOv5 folder
os.chdir('D:/cvprojects/proposal/construction/yolov5')  # Change this to your YOLOv5 directory

# Load the YOLOv5 model and train
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load a pre-trained YOLOv5 model
model.train(data=data_yaml_path, epochs=50)  # Adjust epochs as needed

print("Training complete.")
