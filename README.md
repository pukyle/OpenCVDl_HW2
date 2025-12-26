# OpenCVDl Homework 2 – Deep Learning with PyTorch & GUI

## Overview
This project implements a **Deep Learning image classification application** using **Python (PyTorch + OpenCV + PyQt5)**.  
It provides a user-friendly GUI to demonstrate two classic neural network architectures: **LeNet-5** (for MNIST) and **ResNet-18** (for CIFAR-10).

The project includes both training scripts (`train.py`) to train models from scratch and a GUI (`main.py`) for visualization, model summary, and real-time inference on custom images.

---

## Features

### 1. **MNIST Classification (LeNet-5)**
- **1.1 Model Architecture** Displays the LeNet-5 structure using `torchsummary`, showing layer details and parameter counts.
- **1.2 Training Visualization** Compares the training accuracy and loss curves between **Sigmoid** and **ReLU** activation functions.
- **1.3 Inference** - Loads a handwritten digit image (0-9).
  - Preprocesses the image (grayscale, resize, invert, normalize).
  - Predicts the digit and displays the probability distribution histogram.
  - *Optimized to correctly distinguish difficult cases (e.g., 6 vs 9).*

---

### 2. **CIFAR-10 Classification (ResNet-18)**
- **2.1 Load Image** Loads and resizes images from the CIFAR-10 dataset or custom sources.
- **2.2 Model Architecture** - Implements a **Modified ResNet-18** specifically optimized for small images (32x32).
  - **Key Modifications:** - First Conv layer: 7x7 (stride 2) → **3x3 (stride 1)**.
    - Removed the first MaxPool layer to preserve spatial features.
    - FC layer output changed to 10 classes.
- **2.3 Accuracy & Loss** Displays the training and validation accuracy/loss curves over 50+ epochs.
- **2.4 Inference** - Classifies images into 10 categories (Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck).
  - **"Others" Detection:** Implements a probability threshold (e.g., < 0.5) to identify images that do not belong to the 10 classes (displays "Predicted: Others").
  - Uses improved normalization and learning rate scheduling for high accuracy (>90%).

---

## GUI Interface
The GUI is built using **PyQt5**, providing an intuitive layout:
- **Left Panel:** Functional buttons for Q1 (LeNet) and Q2 (ResNet).
- **Right Panel:** Image display area for input images.
- **Pop-ups:** Matplotlib charts for Loss/Accuracy curves and Probability histograms.
- **Status Label:** Shows the predicted class and confidence score at the bottom.

---

## Project Structure
```text
OpenCV_HW2/
├── Hw2/
│   ├── main.py              # Main GUI application
│   ├── train.py             # Training script for LeNet and ResNet
│   ├── model/               # Saved model weights (.pth files)
│   │   ├── Weight_Relu.pth
│   │   └── weight.pth (ResNet)
│   ├── data/                # Dataset folders (MNIST / CIFAR-10)
│   └── Q2_inference_img/    # Test images
├── venv/                    # Python Virtual Environment (optional)
└── README.md
```
## Requirements
* Python 3.10+
* PyTorch (GPU/CUDA version recommended)
* Torchvision
* OpenCV (cv2)
* PyQt5
* Matplotlib
* Torchsummary

Install all dependencies:

```Bash
# For CUDA 11.8 (Recommended for NVIDIA GPU)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install opencv-contrib-python matplotlib PyQt5 torchsummary
```
## Run the Program
### 1. Train the Models (Optional)
If you need to retrain the models from scratch:

```Bash
cd Hw2
python train.py
```
This will generate .pth files in the model/ directory.
### 2. Run the GUI
To start the application:
```Bash
cd Hw2
python main.py
```
The GUI window will appear.
* Q1: Load an image of a number and click "Predict".
* Q2: Load a CIFAR-10 image (or an "Others" image) and click "Inference".
