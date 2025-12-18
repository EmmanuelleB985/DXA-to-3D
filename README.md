# 3D Spine Shape Estimation from Single 2D DXA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CI/CD](https://github.com/EmmanuelleB985/DXA-to-3D/actions/workflows/ci.yml/badge.svg)](https://github.com/EmmanuelleB985/DXA-to-3D/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**[Project Page](https://www.robots.ox.ac.uk/~vgg/research/dxa-to-3d/) | [Paper](https://www.robots.ox.ac.uk/~vgg/research/dxa-to-3d/paper.pdf) | [Dataset](#dataset)**

## Overview

This repository contains the official implementation of **"3D Spine Shape Estimation from Single 2D DXA"** (MICCAI 2024 Oral).

We present an automated framework to estimate 3D spine shapes from 2D DXA scans, enabling patient-specific understanding of spinal deformations including scoliosis. Our method predicts coronal and sagittal spine curves from a single AP DXA scan, allowing for 3D spine reconstruction.

### Key Contributions

1. **Novel Regression Framework**: First method to regress 3D patient-specific spine shapes from 2D AP DXA only
2. **Lightweight Architecture**: Efficient transformer and ResNet50 backbone surpassing complex models
3. **Clinical Applicability**: User-friendly 3D visualization for scoliosis measurement
4. **Comprehensive Evaluation**: Validated on UK Biobank paired DXA-MRI dataset

## Key Features

- **Multiple Model Architectures**: Transformer and ResNet50 models
- **Comprehensive Metrics**: Angle, curvature, and 3D reconstruction metrics
- **Advanced Visualization**: 2D projections and 3D mesh generation
- **Efficient Training**: Mixed precision training, gradient accumulation
- **Production Ready**: Type hints, comprehensive testing, CI/CD pipeline
- **Well Documented**: Google-style docstrings 

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 10GB+ GPU VRAM for training

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/EmmanuelleB985/DXA-to-3D.git
cd DXA-to-3D
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "from model import create_model; print('Installation successful!')"
```

### Docker Installation (Alternative)

```bash
docker build -t dxa-to-3d .
docker run --gpus all -it dxa-to-3d
```

## Dataset

### UK Biobank DXA-MRI Dataset

The paired DXA-MRI data can be obtained from UK Biobank after registration.

1. **Register at UK Biobank**: Create an account at [UK Biobank](https://www.ukbiobank.ac.uk/)
2. **Download preprocessing tools**:
```bash
git clone https://github.com/rwindsor1/UKBiobankDXAMRIPreprocessing
```
3. **Follow preprocessing instructions** in the linked repository

### Dataset Structure

Organize your data as follows:
```
data/
├── train/
│   ├── images/
│   │   ├── patient_001.png
│   │   └── ...
│   └── annotations/
│       ├── patient_001.json
│       └── ...
├── val/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

### Custom Dataset

To use your own dataset, ensure annotations follow this format:
```json
{
  "coronal_centerline": [[x1, y1], [x2, y2], ...],
  "coronal_lateral": [[x1, y1], [x2, y2], ...],
  "sagittal_centerline": [[x1, y1], [x2, y2], ...],
  "sagittal_lateral": [[x1, y1], [x2, y2], ...]
}
```

## Quick Start

### Download Pre-trained Model

```bash
# Download checkpoint
wget https://www.dropbox.com/scl/fi/be4dg1xccgl1fo9wn74i8/epoch-996-loss_valid-points-best_loss-0.0168.pt?rlkey=ytnrrctofyebqtkj5p4554px1 -O checkpoints/best_model.pt
```

### Run Inference

```python
from inference import SpineEstimator
from config import InferenceConfig

# Configure
config = InferenceConfig(
    model_checkpoint='checkpoints/best_model.pt',
    input_dir='test_images/',
    output_dir='outputs/'
)

# Create estimator
estimator = SpineEstimator(config)

# Process single image
results = estimator.predict_single('path/to/dxa_image.png')

# Process directory
results = estimator.predict_directory('test_images/')
```

## Model Architecture

### Transformer Model

```python
from model import SpineTransformer

model = SpineTransformer(
    input_channels=1,
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
    num_points=100,
    dropout=0.1
)
```

### ResNet50 Model

```python
from model import ResNet50SpineModel

model = ResNet50SpineModel(
    input_channels=1,
    num_points=100,
    pretrained=True,
    freeze_backbone=False
)
```

## Training

### Configuration

Create a training configuration file `configs/train_config.yaml`:

```yaml
experiment_name: dxa_spine_transformer
seed: 42
num_epochs: 100
batch_size: 8
num_workers: 4

model:
  type: transformer
  params:
    hidden_dim: 512
    num_heads: 8
    num_layers: 6

optimizer:
  name: adamw
  lr: 1e-4
  weight_decay: 1e-5

data:
  train_dir: /path/to/train
  val_dir: /path/to/val
```

### Start Training

```bash
# Using default configuration
python train.py

# Using custom configuration
python train.py --config configs/train_config.yaml

# With Hydra overrides
python train.py model.type=resnet50 optimizer.lr=5e-5
```

### Monitor Training

Training progress is logged to TensorBoard and optionally Weights & Biases:

```bash
# TensorBoard
tensorboard --logdir outputs/

# Weights & Biases (if enabled)
wandb login
```

## Inference

### Command Line Interface

```bash
# Process single image
python inference.py --input image.png --output results/

# Process directory
python inference.py --input test_images/ --output results/

# With specific checkpoint
python inference.py --checkpoint checkpoints/best_model.pt --input test_images/
```

### Python API

```python
from inference import SpineEstimator
import numpy as np

# Initialize estimator
estimator = SpineEstimator(config)

# Get predictions
predictions = estimator.predict_single('dxa_image.png')

# Access results
coronal_curve = predictions['coronal_centerline']  # (100, 2) array
angle = predictions['angle']  # angle in degrees
spine_3d = predictions['spine_3d']  # (100, 3) 3D coordinates
```

## Evaluation

### Metrics

The following metrics are computed during evaluation:

- **Point-wise Metrics**: MSE, MAE, RMSE
- **Curve Metrics**: Mean/max distance, curvature similarity
- **Clinical Metrics**: Modified Ferguson angle error, symmetry loss
- **3D Reconstruction**: 3D point cloud accuracy

### Run Evaluation

```python
from evaluation import evaluate_model

metrics = evaluate_model(
    model_path='checkpoints/best_model.pt',
    test_data='data/test/',
    output_dir='evaluation_results/'
)
```

## Results

### Quantitative Results

| Model | MSE ↓ | MAE ↓ | Modified Ferguson Angle Error ↓ | 3D Distance ↓ |
|-------|-------|-------|-------------------|---------------|
| ResNet50 | 0.0182 | 0.0134 | 2.3° | 4.2mm |
| **Transformer (Ours)** | **0.0168** | **0.0121** | **1.9°** | **3.8mm** |

## API Documentation

### Core Modules

#### `model.py`
- `SpineTransformer`: Transformer-based model
- `ResNet50SpineModel`: ResNet50-based model
- `SpineShapeLoss`: Custom loss function
- `create_model()`: Model factory function

#### `dataset.py`
- `DXADataset`: Main dataset class
- `DXADataModule`: Data module for managing datasets
- `get_data_transforms()`: Data augmentation pipeline

#### `utils.py`
- `reconstruct_3d_spine()`: 3D reconstruction from projections
- `calculate_cobb_angle()`: Modified Ferguson angle calculation
- `visualize_predictions()`: Visualization utilities
- `save_3d_spine_mesh()`: 3D mesh export

#### `train.py`
- `Trainer`: Main training class
- Training loop with validation
- Checkpointing and early stopping

#### `inference.py`
- `SpineEstimator`: Inference engine
- Batch processing capabilities
- Result visualization and export

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black . --line-length 100
isort . --profile black

# Type checking
mypy . --ignore-missing-imports

# Linting
flake8 . --max-line-length 100
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UK Biobank for providing access to the DXA-MRI dataset

If you use this work in your research, please cite:

```bibtex
@InProceedings{Bou_3D_MICCAI2024,
    author = {Bourigault, Emmanuelle and Jamaludin, Amir and Zisserman, Andrew},
    title = {3D Spine Shape Estimation from Single 2D DXA},
    booktitle = {Proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
    year = {2024},
    publisher = {Springer Nature Switzerland},
    volume = {LNCS 15005},
    month = {October},
    pages = {pending}
}

@InProceedings{Windsor21,
    author = {Rhydian Windsor and Amir Jamaludin and Timor Kadir and Andrew Zisserman},
    booktitle = {Proc. Medical Image Computing and Computer Aided Intervention (MICCAI)},
    title = {Self-Supervised Multi-Modal Alignment for Whole Body Medical Imaging},
    year = {2021}
}
```
