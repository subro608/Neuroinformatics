# LFP2vec Neuroinformatics - Complete Repository Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation & Dependencies](#installation--dependencies)
4. [Dataset Requirements & Preparation](#dataset-requirements--preparation)
5. [Model Architectures](#model-architectures)
6. [Training Guide](#training-guide)
7. [Inference & Testing](#inference--testing)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Troubleshooting](#troubleshooting)
10. [Development Commands](#development-commands)

---

## 🧠 Project Overview

This repository implements state-of-the-art deep learning models for EEG-based neurological disorder classification, specifically targeting:
- **Alzheimer's Disease (AD)**
- **Frontotemporal Dementia (FTD)**  
- **Healthy Controls (CN)**

The project leverages multi-modal EEG analysis combining spatial, temporal, and spectral features through advanced architectures including transformers, graph neural networks, and hybrid models.

### Key Features
- **Multi-scale spectral analysis** across delta, theta, alpha, beta, and gamma bands
- **Graph-based spatial modeling** of electrode relationships
- **Transformer-based temporal processing**
- **Cross-attention fusion mechanisms**
- **Support for both within-subject and cross-subject evaluation**

---

## 📁 Repository Structure

```
D:\LFP2vec\Neuroinformatics\
│
├── datasets/               # Data loading and preprocessing modules
│   ├── data_prep.py       # Main data preparation pipeline
│   ├── eeg_dataset.py     # Basic EEG dataset loader
│   ├── eeg_dataset_adformer.py
│   ├── eeg_dataset_multiscalegraph_spectral_advanced.py
│   ├── eeg_mvt_dataset.py
│   ├── eeg_mvt_dataset_mfeat.py
│   └── eeg_svm_dataset.py
│
├── models/                 # Model implementations
│   ├── eeg_adformer.py    # ADFormer architecture
│   ├── eeg_multi_scale_graph_spectral_advanced.py  # Best performing model
│   ├── eeg_mvt.py         # Multi-View Transformer
│   ├── eeg_mvt_mfeat.py   # MVT with wavelet features
│   ├── eeg_net.py         # EEGNet implementation
│   ├── eeg_svm.py         # SVM classifier
│   ├── model_architecture.py
│   └── best_model_overall_multiscale_graph_spectral.pth  # Pre-trained weights
│
├── training/               # Training scripts
│   ├── train_kfold.py
│   ├── train_kfold_adformer.py
│   ├── train_kfold_multi_scale_graph_spectral_advanced.py
│   ├── train_kfold_mvt.py
│   ├── train_kfold_mvt_mfeat.py
│   ├── train_kfold_svm.py
│   ├── train_kfold_svm_grid.py
│   └── hyperparameter_tuning.py
│
├── testing/                # Evaluation scripts
│   ├── test.py
│   ├── test_adformer.py
│   ├── test_multiscale_graph_spectral_advanced.py
│   ├── test_mvt.py
│   ├── test_mvt_mfeat.py
│   ├── test_svm.py
│   ├── cuda_test.py
│   └── sql_test.py
│
├── visualization/          # Visualization tools
│   ├── viz.py
│   ├── data_vis.py
│   ├── cross_attention_interactive.html
│   └── [various diagrams and images]
│
├── layers/                 # Custom neural network layers
│   ├── ADformer_EncDec.py
│   ├── Augmentation.py
│   ├── Embed.py
│   └── SelfAttention_Family.py
│
├── utils/                  # Utility functions
│   └── masking.py
│
└── data_provider/          # Data loading utilities
    └── uea.py
```

---

## 🔧 Installation & Dependencies

### System Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU acceleration)
- **RAM**: Minimum 16GB (32GB recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Install Dependencies

```bash
# Create virtual environment
python -m venv eeg_env
source eeg_env/bin/activate  # On Windows: eeg_env\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Required Packages

Create a `requirements.txt` file:
```txt
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.3.0
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
mne==1.5.0
pywt==1.4.1
networkx==3.1
wandb==0.15.8
tqdm==4.65.0
joblib==1.3.1
```

### Verify Installation

```bash
# Check CUDA availability
python testing/cuda_test.py

# Should output:
# CUDA Available: True
# CUDA Device: <your GPU name>
```

---

## 📊 Dataset Requirements & Preparation

### Data Format Requirements

#### Input Data Structure
```
eeg_data/
├── participants.tsv        # Subject metadata (ID, Group, Age, etc.)
└── derivatives/
    └── sub-XXX/
        └── eeg/
            └── sub-XXX_task-eyesclosed_eeg.set  # EEGLAB format
```

#### Expected File Format
- **File Type**: EEGLAB `.set` files
- **Sampling Rate**: Will be resampled to 95 Hz
- **Channels**: 19 EEG channels (10-20 system)
- **Recording Type**: Eyes-closed resting state

### Step 1: Download/Prepare Raw Data

1. **Create data directory structure**:
```bash
mkdir -p ../eeg_data/derivatives
```

2. **Prepare participants.tsv**:
```csv
participant_id,Group,Age,Sex
sub-001,A,72,M
sub-002,C,68,F
sub-003,F,75,M
```

Groups: A=Alzheimer's, C=Control, F=Frontotemporal Dementia

3. **Place EEG files** in appropriate directories following the naming convention

### Step 2: Run Data Preparation Pipeline

```bash
cd datasets
python data_prep.py
```

This will:
- **Crop** 30 seconds from start/end
- **Resample** to 95 Hz
- **Chunk** into 15-second segments (1424 samples)
- **Split** into train (80%) and test (20%) sets
- **Create** within-subject test set (10% of training)
- **Save** processed chunks to `model_data/` directory

### Output Structure After Processing
```
model_data/
├── labels.json             # Dataset metadata and splits
├── train/
│   └── sub-XXX_eeg_chunk_0.set
│   └── sub-XXX_eeg_chunk_1.set
│   └── ...
└── test/
    └── sub-YYY_eeg_chunk_0.set
    └── ...
```

### Step 3: Verify Data Integrity

```python
import json

# Check generated labels file
with open('model_data/labels.json', 'r') as f:
    labels = json.load(f)
    
print(f"Total subjects: {len(labels['all_participants'])}")
print(f"Training subjects: {len(labels['train_participants'])}")
print(f"Test subjects: {len(labels['test_participants'])}")
print(f"Total chunks: {len(labels['chunk_labels'])}")
```

---

## 🏗️ Model Architectures

### 1. **Multiscale Graph Spectral Model** (Best Performance: 98.55%)

The flagship model combining spatial and spectral processing:

```python
# Architecture highlights
- Spatial Feature Extractor: CNN + Transformer encoders
- Multi-Scale Graph Module: GCN layers at multiple frequency scales
- Cross-Attention Fusion: Bidirectional attention between modalities
- Dimension: 264 (best), 128 (lighter version)
```

### 2. **Multi-View Transformer (MVT)**

```python
# Features
- Separate encoders for time and frequency domains
- Cross-modal attention mechanism
- Position embeddings for temporal context
```

### 3. **ADFormer**

```python
# Attention-based architecture
- Transformer encoder-decoder structure
- Data augmentation during training
- Suitable for variable-length sequences
```

### 4. **EEGNet**

```python
# Compact CNN architecture
- Depthwise separable convolutions
- Temporal and spatial filters
- Lightweight and efficient
```

### 5. **SVM Classifier**

```python
# Traditional ML approach
- Hand-crafted features
- Statistical and spectral features
- Grid search optimization
```

---

## 🚀 Training Guide

### Basic Training Command

```bash
# Navigate to training directory
cd training

# Train the best performing model
python train_kfold_multi_scale_graph_spectral_advanced.py
```

### Training Configuration

Edit training scripts to modify parameters:

```python
# Key hyperparameters
num_channels = 19
num_classes = 3
dim = 264           # Model dimension (64, 128, or 264)
scales = [3, 4, 5]  # Multi-scale factors
epochs = 200
learning_rate = 0.0003
batch_size = 16     # Adjust based on GPU memory
weight_decay = 0.001
```

### Training Different Models

```bash
# Multi-View Transformer
python train_kfold_mvt.py

# Multi-View Transformer with Wavelet
python train_kfold_mvt_mfeat.py

# ADFormer
python train_kfold_adformer.py

# EEGNet
python train_kfold.py

# SVM with Grid Search
python train_kfold_svm_grid.py
```

### Monitor Training with Weights & Biases

```python
# Set up W&B (optional)
wandb login
# Training scripts automatically log to W&B if available
```

### Resume Training from Checkpoint

Most training scripts support checkpoint resuming:

```python
# Checkpoints are saved in checkpoint/ directory
# Scripts automatically detect and resume from latest checkpoint
```

---

## 🔮 Inference & Testing

### Quick Inference with Pre-trained Model

```bash
cd testing

# Test with the best pre-trained model
python test_multiscale_graph_spectral_advanced.py \
    --model_path "../models/best_model_overall_multiscale_graph_spectral.pth" \
    --data_dir "../model_data" \
    --device "cuda" \
    --batch_size 16
```

### Test Different Models

```bash
# Test MVT model
python test_mvt.py

# Test ADFormer
python test_adformer.py

# Test SVM
python test_svm.py
```

### Custom Inference Script

```python
import torch
from models.eeg_multi_scale_graph_spectral_advanced import create_model
from datasets.eeg_dataset_multiscalegraph_spectral_advanced import SpatialSpectralEEGDataset

# Load model
model = create_model(
    num_channels=19,
    num_classes=3,
    dim=264,
    scales=[3, 4, 5],
    time_length=1424
)

# Load checkpoint
checkpoint = torch.load('models/best_model_overall_multiscale_graph_spectral.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load your data
dataset = SpatialSpectralEEGDataset(
    data_dir='model_data/test',
    labels_file='model_data/labels.json',
    scales=[3, 4, 5]
)

# Run inference
with torch.no_grad():
    for data, label in dataset:
        raw_eeg = data['raw_eeg'].unsqueeze(0)
        scale_inputs = {k: v.unsqueeze(0) for k, v in data.items() if k != 'raw_eeg'}
        
        output = model(raw_eeg, scale_inputs)
        prediction = output.argmax(dim=1)
        
        print(f"True: {label}, Predicted: {prediction.item()}")
```

### Batch Processing

```python
from torch.utils.data import DataLoader

# Create dataloader for batch processing
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# Process in batches
predictions = []
for batch_data, batch_labels in dataloader:
    with torch.no_grad():
        outputs = model(batch_data['raw_eeg'], scale_inputs)
        preds = outputs.argmax(dim=1)
        predictions.extend(preds.cpu().numpy())
```

---

## 📈 Performance Benchmarks

### Model Comparison Table

| Model | Params | Within-Subject | Cross-Subject | Training Time | Inference Speed |
|-------|--------|----------------|---------------|---------------|-----------------|
| **Multiscale GS (264)** | 12.3M | **98.55%** | 59.68% | ~8h | 45ms/batch |
| Multiscale GS (128) | 5.8M | 97.38% | 63.46% | ~4h | 25ms/batch |
| MVT | 8.2M | 89.83% | 64.95% | ~3h | 30ms/batch |
| MVT + Wavelet | 9.1M | 90.69% | 63.80% | ~3.5h | 35ms/batch |
| ADFormer | 6.5M | 85.20% | 58.30% | ~2.5h | 20ms/batch |
| EEGNet | 2.1M | 82.15% | 55.60% | ~1h | 10ms/batch |
| SVM | N/A | 78.50% | 52.30% | ~30min | 5ms/sample |

### Binary Classification Performance

| Task | Multiscale GS | MVT | SVM |
|------|---------------|-----|-----|
| AD vs CN | 98.53% | 87.13% | 75.20% |
| FTD vs CN | 98.99% | 93.94% | 82.10% |
| AD vs FTD | 98.17% | 89.45% | 71.50% |

---

## 🔨 Development Commands

### Code Formatting

```bash
# Format with black
python -m black . --line-length 88

# Sort imports
python -m isort . --profile black

# Run linting
python -m flake8 . --max-line-length 88
```

### Run Tests

```bash
# Test CUDA setup
python testing/cuda_test.py

# Run unit tests (if available)
python -m pytest tests/
```

### Generate Visualizations

```bash
cd visualization

# Generate performance plots
python viz.py --model_results ../results/

# Create confusion matrices
python data_vis.py --predictions ../outputs/predictions.csv
```

### Hyperparameter Optimization

```bash
cd training

# Run hyperparameter search
python hyperparameter_tuning.py \
    --model "multiscale_gs" \
    --trials 50 \
    --gpu 0
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **CUDA Out of Memory**
```python
# Reduce batch size in training script
batch_size = 8  # or 4 for very limited memory

# Enable gradient accumulation
accumulation_steps = 4

# Use mixed precision training
scaler = torch.cuda.amp.GradScaler()
```

#### 2. **MNE Import Error**
```bash
# Install with CUDA support
pip install mne[cuda]

# Or install specific version
pip install mne==1.5.0
```

#### 3. **Data Loading Issues**
```python
# Check file paths
import os
data_path = os.path.abspath('../eeg_data')
print(f"Looking for data in: {data_path}")

# Verify file format
import mne
raw = mne.io.read_raw_eeglab('your_file.set', preload=True)
print(raw.info)
```

#### 4. **Training Instability**
```python
# Adjust learning rate
learning_rate = 0.0001  # Lower for stability

# Increase gradient clipping
max_grad_norm = 0.5

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

#### 5. **Poor Cross-Subject Performance**
```python
# Add more augmentation
augmentation_prob = 0.5  # Increase augmentation

# Use domain adaptation techniques
# Consider pre-training on larger dataset
```

---

## 📝 Command Reference

### Essential Commands for Quick Start

```bash
# 1. Prepare environment
python -m venv eeg_env
source eeg_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data
cd datasets
python data_prep.py

# 4. Train best model
cd ../training
python train_kfold_multi_scale_graph_spectral_advanced.py

# 5. Test model
cd ../testing
python test_multiscale_graph_spectral_advanced.py

# 6. Visualize results
cd ../visualization
python viz.py
```

### Advanced Usage

```bash
# Custom training with specific parameters
python train_kfold_multi_scale_graph_spectral_advanced.py \
    --dim 128 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.0005

# Inference on custom data
python test_multiscale_graph_spectral_advanced.py \
    --data_path "/path/to/your/data" \
    --model_path "custom_model.pth" \
    --output_dir "predictions/"

# Hyperparameter search
python hyperparameter_tuning.py \
    --config "configs/search_space.yaml" \
    --n_trials 100
```

---

## 📞 Contact & Support

For issues, questions, or contributions:
1. Check existing documentation
2. Review code comments and docstrings
3. Examine similar implementations in the codebase
4. Refer to original papers for algorithmic details

---

## 📄 License & Citation

If using this code for research, please cite:
```bibtex
@software{lfp2vec_neuroinformatics,
  title = {LFP2vec Neuroinformatics: Multi-modal EEG Analysis for Neurological Disorder Classification},
  year = {2024},
  url = {https://github.com/yourusername/LFP2vec}
}
```

---

## 🎯 Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Set up virtual environment
- [ ] Install PyTorch with CUDA support
- [ ] Install required packages
- [ ] Prepare EEG data in correct format
- [ ] Run data preparation script
- [ ] Choose model to train
- [ ] Start training
- [ ] Evaluate on test set
- [ ] Visualize results

---

*Last Updated: February 2025*