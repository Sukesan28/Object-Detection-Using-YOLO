# YOLOv8 Object Detection (Python + OpenCV + ultralytics)


## 📌 Project Overview
This project demonstrates **real-time object detection** using **YOLOv8**.  
It can detect multiple objects in images, videos, or webcam streams, annotate them with bounding boxes, and optionally announce detected objects.

---

## ✅ Key Features & Concepts Used

### 🧠 Python & OpenCV
- ✅ Python Fundamentals: Classes, functions, loops, and conditionals
- 🧩 OpenCV: Used to process images/videos and display annotated frames
- 🔍 Object Detection: Real-time detection using YOLOv8
- ⚡ Performance: Confidence threshold filtering to detect only high-probability objects

### 💾 Model & Detection
- YOLOv8 Pretrained Model (`yolov8n.pt`)
- Detects multiple object classes from COCO dataset
- Annotates frames with bounding boxes and class labels

### 🔊 Optional Audio Feedback
- Announce newly detected objects using **gTTS** (works in Jupyter/Colab)
- Avoids repeated announcements for the same object

---

## 🛠 Tools & Technologies

| Category        | Tools/Technologies               |
|-----------------|---------------------------------|
| Language        | Python                          |
| Framework       | ultralytics YOLOv8              |
| Image/Video     | OpenCV                          |
| Audio (Optional)| gTTS                            |
| IDE             | Jupyter Notebook / VS Code      |

---

## 📂 Folder Structure
```bash
YOLOv8-Object-Detection/
│
├── src/
│   ├── Yolo_pytorch.py           # YOLOv8 detection class
│   ├── detect_video.py           # Script to run detection on videos
│   └── detect_webcam.py          # Script to run detection via webcam
│
├── models/
│   └── yolov8n.pt                # Pretrained YOLOv8 model
│
├── videos/
│   └── sample_video.mp4          # Example video
│
└── README.md
