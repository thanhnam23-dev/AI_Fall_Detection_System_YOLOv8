# AI Fall Detection System using YOLOv8

## Requirements

- Python 3.11
- Miniconda (recommended)
- NVIDIA GPU (optional)

---

## Clone repository

```bash
git clone https://github.com/thanhnam23-dev/AI_Fall_Detection_System_YOLOv8.git
cd AI_Fall_Detection_System_YOLOv8
```

---

## Create Conda Environment

```bash
conda create -n yolo_fall python=3.11
conda activate yolo_fall
```

---

## Install PyTorch

Install the correct PyTorch version for your system from:

https://pytorch.org/get-started/locally/

Example (CUDA 12.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Example (CPU only):

```bash
pip install torch torchvision torchaudio
```

---

## Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## Verify Installation

```bash
yolo checks
```

Expected output:

```
Ultralytics 8.x.x
CUDA: True
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

---

## Run

Example:

```bash
python detect.py
```