# AI Surveillance System

An end-to-end AI surveillance system that tracks and identifies people in video streams. This project integrates face recognition, object detection, person re-identification, and multi-object tracking to deliver robust person tracking capabilities.
## Output Sample
![Demo](assets/output.gif)

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Model Details](#model-details)
6. [Contributing](#contributing)
7. [License](#license)
8. [References](#references)

---

## Features

* **Face Re-identification**: Fine-tuned FaceNet for generating robust face embeddings (used independently from tracking).
* **Object Detection**: Fine-tuned YOLOv11m for accurate person detection in video frames.
* **Person Re-identification**: Fine-tuned OSNet to distinguish individuals across frames.
* **Multi-Object Tracking**: Combines YOLOv11m detections, BoTSORT tracker, and OSNet embeddings to maintain consistent IDs over time.

---

## Architecture Overview

```text
┌──────────────┐      ┌─────────────┐      ┌───────────┐
│ Video Stream │ ───► │ Object      │ ───► │ Tracking  │
│              │      │ Detection   │      │ Pipeline  │
└──────────────┘      └─────────────┘      └───────────┘
                              │               │
                              │               │
                              ▼               │
                        ┌─────────────┐       │
                        │ Embedding   │       │
                        │ Extraction  │       │
                        │ (OSNet)     │◄──────┘
                        └─────────────┘       │
                              │               │
                              ▼               │
                        ┌───────────┐          
                        │ BoTSORT   │          
                        │ Tracker   │          
                        └───────────┘          
                              │               
                              ▼               
                  ┌────────────────────┐      
                  │ ID-consistent      │      
                  │ Bounding Boxes     │      
                  └────────────────────┘      
```

* **Object Detection**: YOLOv11m locates people in each frame.
* **Embedding Extraction**: OSNet produces appearance embeddings for each detected person.
* **Tracking**: BoTSORT uses motion and OSNet embeddings to associate detections across frames.
* **Face Re-identification** (separate): FaceNet can be run independently on cropped faces when needed.

---

## Project Structure

```
├── Face Re-Identification
│   ├── Face Re-Identification.py
│   ├── dataset
│   │   ├── eval_set.csv
│   │   ├── test
│   │   ├── train
│   │   └── trainset.csv
│   └── facenet-finetuning.py
├── Object Detection
│   └── yolo-v11-training.ipynb
├── ReID Model
│   └── reid-training-final.ipynb
├── Tracking
    └── Tracking.py
```

---

## How to Run
 
### 1. Clone the repository
 
```bash
git clone https://github.com/abdelrahman-omran/AI-Surveillance-System.git
cd AI-Surveillance-System
```
 
### 2. Create and activate a virtual environment
 
```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Download model weights
 
Place your fine-tuned model weights in the appropriate directories:
- YOLOv11m weights → `Object Detection/`
- OSNet weights → `ReID Model/`
- FaceNet weights → `Face Re-Identification/`
 
### 5. Run the tracker
 
```bash
python Tracking/Tracking.py
```
 
### 6. Run face re-identification (optional, standalone)
 
```bash
python "Face Re-Identification/Face Re-Identification.py"
```
 
### 6. Train your own models (optional)
 
Open and run the notebooks in Jupyter:
 
```bash
jupyter notebook
```
 
- `Object Detection/yolo-v11-training.ipynb` — fine-tune YOLOv11m
- `ReID Model/reid-training-final.ipynb` — fine-tune OSNet
- `Face Re-Identification/facenet-finetuning.py` — fine-tune FaceNet
 
---
---

## Model Details

| Component                | Purpose                  | Backbone                    | Pre-trained on | Fine-tuned Dataset       | Output                                |
| ------------------------ | ------------------------ | --------------------------- | -------------- | ------------------------ | ------------------------------------- |
| Face Re-identification   | Face feature extraction  | InceptionResnetV1 (FaceNet) | VGGFace2       | Your face dataset        | 512-d face embeddings                 |
| Object Detection         | Person localization      | YOLOv11m                    | COCO           | Custom surveillance data | Bounding boxes + confidence scores    |
| Person Re-identification | Appearance embedding     | OSNet                       | Market-1501    | Custom tracklets         | 512-d appearance embeddings           |
| Tracking                 | Multi-object association | BoTSORT + OSNet embeddings  | N/A            | N/A                      | ID-consistent bounding boxes in video |

---

## Contributing

Contributions are welcome! Please open issues and submit pull requests for bug fixes and feature enhancements.

---

## License

This project is licensed under the Apache License 2.0  License. See [LICENSE](LICENSE) for details.

---

## References

* FaceNet: [https://arxiv.org/abs/1503.03832](https://arxiv.org/abs/1503.03832)
* YOLOv11: [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)
* OSNet: [https://arxiv.org/abs/1905.00953](https://arxiv.org/abs/1905.00953)
* BoTSORT: [https://arxiv.org/abs/2206.14651](https://arxiv.org/abs/2206.14651)
