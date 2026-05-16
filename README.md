# drone-human-detection-yolov11
AI/ML-based drone image human and car detection system with human counting using YOLOv11.
# Drone Human Detection & Counting — YOLOv11 × VisDrone 2019
Fine-tuned YOLOv11 on the VisDrone 2019 aerial dataset to detect humans, cars, and other objects from drone imagery.
# Dataset
VisDrone 2019 DET — drone footage with 10 object classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.
Source: Kaggle – VisDrone Dataset

# What I Did
# Dataset Understanding

Explored folder structure and YAML config
Sanity checked image-label pairs across train, val, and test splits
Analyzed class distribution, bounding box size, aspect ratio, and object density per split
Visualized sample images from all 3 splits

# Preprocessing

Parsed YOLO-format labels (5-column: class, x, y, w, h)
Computed class weights using sklearn to handle class imbalance

# Model Training

Fine-tuned YOLOv11 using Ultralytics on Kaggle GPU
Input size: 640×640

# Evaluation

Validated on test split using model.val()
Reported mAP50, mAP50-95, Precision, Recall

# Detection & Counting

Detected humans (pedestrian + people) and cars separately
Drew bounding boxes with class labels
Displayed total human count on each image

# Visualization

Training/validation loss curves (box, class, DFL)
Precision, Recall, mAP50, mAP50-95 over epochs
Train vs val loss comparison
Precision vs Recall scatter plot
Loss component stacked area chart
Metrics correlation heatmap
Training time per epoch
7×7 grid of test predictions


# Installation
bashpip install ultralytics opencv-python pandas matplotlib seaborn scikit-learn tqdm pillow pyyaml

# Project Structure
├── Protiva_Project_YoloV11.ipynb
├── visdrone.yaml
└── runs/detect/train/weights/
    ├── best.pt
    └── last.pt

