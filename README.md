# 🏃‍♂️ GCN Action Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview
This repository contains the source code for my Action Recognition System. The project leverages **Graph Convolutional Networks (GCNs)**, specifically focusing on Spatial-Temporal configurations (ST-GCN), to classify and recognize human behaviors based on skeleton data.

Unlike traditional CNN-based approaches that process raw RGB frames, this system models the human body as a graph (where joints are nodes and bones are edges), making it highly efficient and robust to background noise.

## ✨ Features
- **Skeleton-based Recognition:** Extracts and processes keypoints/skeleton data for accurate pose representation.
- **ST-GCN Architecture:** Implements Spatial-Temporal Graph Convolutional Networks to capture both spatial configurations of joints and their temporal dynamics across frames.
- **Real-time Processing (Optional):** Capable of inferring actions from live camera feeds or pre-recorded videos.
- **Custom Dataset Support:** Scripts included to train the model on custom action datasets.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AnhPhong89/gcn-action-recognition.git](https://github.com/AnhPhong89/gcn-action-recognition.git)
   cd gcn-action-recognition
