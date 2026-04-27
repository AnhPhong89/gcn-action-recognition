# 🏃‍♂️ GCN Action Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview
This repository contains the source code for an Action Recognition System. The project leverages **Graph Convolutional Networks (GCNs)**, specifically focusing on Spatial-Temporal configurations (ST-GCN), to classify and recognize human behaviors based on skeleton data.

Unlike traditional CNN-based approaches that process raw RGB frames, this system models the human body as a graph (where joints are nodes and bones are edges), making it highly efficient, robust to background noise, and computationally lighter.

Current supported classes: `falling`, `sitting`, `standing`, `walking`, `walking_while_using_phone`.

## ✨ Features
- **Video to Skeleton Pipeline:** Automatically extracts COCO-17 keypoints from raw `.mp4` or `.avi` videos using `YOLOv11n-pose`.
- **ST-GCN Architecture:** Implements Spatial-Temporal Graph Convolutional Networks with dynamic graph layouts.
- **Robust Training Engine:** Includes Mixed Precision (AMP), TensorBoard logging, CSV history logging, and advanced loss functions (Focal Loss, Label Smoothing).
- **Automated Data Processing:** Scripts to handle temporal padding/truncating, spatial normalization (hip-centric), and temporal/spatial augmentations.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnhPhong89/gcn-action-recognition.git
   cd gcn-action-recognition
   ```

2. **Install dependencies:**
    It is recommended to use an Anaconda environment or Python virtualenv.
   ```bash
   pip install -r requirements.txt
   pip install ultralytics # for YOLOv11 pose extraction
   ```

---

## 🚀 Usage Guide

### 1. Data Preparation & Preprocessing
Place your raw videos inside the `data/raw/` directory, categorized by class folder:
```text
data/raw/
├── falling/
├── sitting/
├── standing/
├── walking/
└── walking_while_using_phone/
```

Run the preprocessing script to extract skeleton coordinates and save them as ST-GCN ready NumPy arrays (`.npy`):

```bash
# Test with a small limit first (2 videos per class)
python scripts/preprocess.py --limit 2

# Process the entire dataset
python scripts/preprocess.py
```
*Output files (`train_data.npy`, `val_data.npy`, etc.) will be saved in `data/processed/`.*

### 2. Training the Model
You can configure hyperparameters (batch size, learning rate, epochs) by editing `configs/base.yaml`.

Start training:
```bash
# Standard training run
python scripts/train.py

# Quick debug run focusing on the first 100 samples
python scripts/train.py --debug

# Resume training from the latest checkpoint
python scripts/train.py --resume runs/exp/checkpoints/checkpoint.pt
```

### 3. Monitoring & Checkpoints
During training, the model saves checkpoints to `runs/exp/checkpoints/`:
- `checkpoint.pt`: Overwritten every epoch (use this to resume if training stops).
- `best.pt`: Saved only when validation accuracy improves.

You can monitor loss, accuracy, and learning rate in real-time using TensorBoard:
```bash
tensorboard --logdir runs/exp/tensorboard
```

---

## 📁 Repository Structure

```text
gcn-action-recognition/
├── configs/
│   └── base.yaml                 # Master configuration file (data, model, training)
├── data/
│   ├── raw/                      # Put your raw .mp4/.avi videos here
│   └── processed/                # Auto-generated .npy and .pkl dataset files
├── scripts/
│   ├── preprocess.py             # Pipeline: Video -> YOLO Pose -> .npy dataset
│   └── train.py                  # Training entry point
├── src/
│   ├── data/                     # DataLoaders, Dataset logic, Augmentations, YOLO wrapper
│   ├── models/                   # ST-GCN Network Architecture definitions
│   ├── training/                 # Trainer loop, Custom Losses, LR Schedulers
│   └── utils/                    # Checkpointing logic, logging, random seed setting
└── README.md
```
