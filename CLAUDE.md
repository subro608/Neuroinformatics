# LFP2vec Neuroinformatics - Complete Repository Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Environment & Singularity Container](#environment--singularity-container)
4. [Dataset Download & Preparation](#dataset-download--preparation)
5. [Model Architectures](#model-architectures)
6. [Multi-GPU Training (SLURM)](#multi-gpu-training-slurm)
7. [Training Guide](#training-guide)
8. [Inference & Testing](#inference--testing)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Known Issues](#known-issues)
11. [Troubleshooting](#troubleshooting)
12. [Changelog](#changelog)

---

## Project Overview

This repository implements state-of-the-art deep learning models for EEG-based neurological disorder classification, specifically targeting:
- **Alzheimer's Disease (AD)** - Group label: `A`
- **Frontotemporal Dementia (FTD)** - Group label: `F`
- **Healthy Controls (CN)** - Group label: `C`

The project leverages multi-modal EEG analysis combining spatial, temporal, and spectral features through advanced architectures including transformers, graph neural networks, and hybrid models.

### Key Features
- Multi-scale spectral analysis across delta, theta, alpha, beta, and gamma bands
- Graph-based spatial modeling of electrode relationships
- Transformer-based temporal processing
- Cross-attention fusion mechanisms
- Support for both within-subject and cross-subject evaluation
- Multi-GPU training via DataParallel on SLURM (NYU HPC)

---

## Repository Structure

```
/scratch/sd5963/Neuroinformatics/
|
├── datasets/               # Data loading and preprocessing modules
│   ├── data_prep.py       # Main data preparation pipeline
│   ├── eeg_dataset.py     # Basic EEG dataset loader
│   ├── eeg_dataset_adformer.py
│   ├── eeg_dataset_multiscalegraph_spectral_advanced.py
│   ├── eeg_mvt_dataset.py
│   ├── eeg_mvt_dataset_mfeat.py
│   ├── eeg_svm_dataset.py
│   ├── intermediate_data/  # Intermediate processing output (from data_prep.py)
│   └── model_data/         # Final processed data with labels.json
│       ├── labels.json
│       ├── train/          # Training chunks (.set files)
│       └── test/           # Test chunks (.set files)
│
├── training/               # Training scripts
│   ├── train_kfold_multi_scale_graph_spectral_advanced.py  # Best model trainer
│   ├── run_train_multigpu.sbatch   # SLURM multi-GPU training script
│   ├── train_kfold.py
│   ├── train_kfold_adformer.py
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
│   └── cross_attention_interactive.html
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
├── data_provider/          # Data loading utilities
│   └── uea.py
│
├── eeg_data/               # Raw dataset (downloaded from OpenNeuro)
│   ├── participants.tsv    # Subject metadata (88 subjects)
│   ├── derivatives/        # Contains .set files per subject
│   │   └── sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
│   ├── dataset_description.json
│   └── sub-XXX/eeg/       # Empty BIDS-format directories
│
├── download_dataset.py     # DEPRECATED: Mendeley download (404)
├── download_openneuro.py   # Working dataset downloader (S3/boto3)
├── CLAUDE.md               # This file
└── README.md
```

---

## Environment & Singularity Container

### Singularity Container

All scripts run inside a Singularity container on NYU HPC:

```
/scratch/sd5963/containers/w2v2_cuda_cu128.sif
```

| Package | Version |
|---------|---------|
| Python | 3.x (use `python3`) |
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.8 |
| wandb | 0.24.0 |
| scikit-learn | 1.7.2 |
| MNE | 1.9.0 |
| pandas | 1.5.3 |
| DDP | Available |

**Important**: The container does NOT have `eeglabio`. For data preparation (exporting .set files), use:

```bash
singularity exec --writable-tmpfs /scratch/sd5963/containers/w2v2_cuda_cu128.sif \
    bash -c "pip install eeglabio -q && python3 <script>"
```

For training and testing, `eeglabio` is NOT needed:

```bash
singularity exec --nv /scratch/sd5963/containers/w2v2_cuda_cu128.sif python3 <script>
```

### SLURM Account & Partition

| Setting | Value |
|---------|-------|
| Account | `torch_pr_60_tandon_advanced` |
| Partition | `l40s_public` |
| GPU | NVIDIA L40S (48GB VRAM each) |
| Log directory | `/scratch/sd5963/slurm_logs/` |

---

## Dataset Download & Preparation

### Source Dataset

**OpenNeuro ds004504** - "A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects"

| Metric | Value |
|--------|-------|
| Source | OpenNeuro S3: `s3://openneuro.org/ds004504/` |
| Subjects | 88 (36 AD, 23 FTD, 29 CN) |
| Format | EEGLAB `.set` files |
| Channels | 19 EEG channels (10-20 system) |
| Recording | Eyes-closed resting state |
| Raw size | ~4.5 GB |
| Location | `/scratch/sd5963/Neuroinformatics/eeg_data/` |

### Step 0: Download Raw Data

The dataset is downloaded from OpenNeuro's S3 bucket using `download_openneuro.py` (unsigned boto3 access):

```bash
cd /scratch/sd5963/Neuroinformatics
python3 download_openneuro.py
```

This downloads all 88 subjects' `.set` files into `eeg_data/derivatives/sub-XXX/eeg/`.

**Note**: The original Mendeley URL (`pbc979p2jc`) is dead (404). Do NOT use `download_dataset.py`.

### Step 1: Data Preparation

Run `data_prep.py` inside the Singularity container (needs `eeglabio`):

```bash
cd /scratch/sd5963/Neuroinformatics/datasets
singularity exec --writable-tmpfs /scratch/sd5963/containers/w2v2_cuda_cu128.sif \
    bash -c "pip install eeglabio -q && python3 data_prep.py"
```

**What it does:**
1. Reads `participants.tsv` and splits subjects 80/20 into train/test
2. Copies `.set` files to `intermediate_data/{train,test}/`
3. Crops 30 seconds from start/end of each recording
4. Resamples to 95 Hz
5. Chunks into 15-second segments (1424 timepoints)
6. 10% of training chunks randomly moved to within-subject test set
7. Saves all chunks + `labels.json` to `model_data/`

### Data Split Results

| Set | AD (A) | CN (C) | FTD (F) | Total |
|-----|--------|--------|---------|-------|
| Train | 1,388 | 1,102 | 729 | 3,219 |
| Test (cross-subject) | 319 | 307 | 247 | 873 |
| Test (within-subject) | 146 | 126 | 72 | 344 |
| **Total** | **1,853** | **1,535** | **1,048** | **4,436** |

- **70 subjects** in train, **18 subjects** in cross-subject test
- Each chunk: 19 channels x 1424 timepoints (15 seconds @ 95 Hz)

### Output Files

```
datasets/model_data/
├── labels.json                          # Array of {file_name, label, type, num_channels, timepoints}
├── train/sub-XXX_eeg_chunk_0.set       # Training chunks
├── train/sub-XXX_eeg_chunk_1.set
├── test/sub-YYY_eeg_chunk_0.set        # Cross-subject test chunks
└── test/sub-ZZZ_eeg_chunk_0.set        # Within-subject test chunks (type: "test_within")
```

**labels.json entry format:**
```json
{
    "file_name": "train/sub-077_eeg_chunk_0.set",
    "label": "F",
    "type": "train",
    "num_channels": 19,
    "timepoints": 1424,
    "total_time (in seconds)": 14.989473684210527
}
```

Types: `"train"`, `"test_cross"`, `"test_within"`

---

## Model Architectures

### 1. Multiscale Graph Spectral Model (Best: 98.55% within-subject)

The flagship model combining spatial and spectral processing:
- **Spatial Feature Extractor**: CNN + Transformer encoders on raw EEG
- **Multi-Scale Graph Module**: GCN layers at multiple frequency scales (delta, theta, alpha, beta, gamma)
- **Cross-Attention Fusion**: Bidirectional attention between spatial and spectral pathways
- **Dimensions**: 264 (best accuracy), 128 (lighter), 64 (fastest)

**Files:**
- Model: `eeg_multi_scale_graph_spectral_advanced.py` (**MISSING - see Known Issues**)
- Dataset: `datasets/eeg_dataset_multiscalegraph_spectral_advanced.py`
- Training: `training/train_kfold_multi_scale_graph_spectral_advanced.py`
- Testing: `testing/test_multiscale_graph_spectral_advanced.py`

### 2. Multi-View Transformer (MVT)
- Separate encoders for time and frequency domains
- Cross-modal attention mechanism
- Best cross-subject accuracy (64.95%)

### 3. MVT with Wavelet Features
- MVT enhanced with wavelet-based features (pywt)

### 4. ADFormer
- Transformer encoder-decoder with augmentation
- Uses custom layers from `layers/` directory

### 5. EEGNet
- Compact CNN with depthwise separable convolutions

### 6. SVM Classifier
- Traditional ML baseline with hand-crafted features

---

## Multi-GPU Training (SLURM)

### SBATCH Script

**Location**: `training/run_train_multigpu.sbatch`

#### SLURM Resources

| Resource | Value |
|----------|-------|
| Partition | `l40s_public` |
| GPUs | 4x L40S (48GB VRAM each) |
| Memory | 128GB |
| CPUs | 32 |
| Time limit | 12 hours |
| Account | `torch_pr_60_tandon_advanced` |

#### Submit Commands

```bash
# Default (4 GPUs, 200 epochs, dim=64, batch_size=32)
sbatch training/run_train_multigpu.sbatch

# Custom settings via environment variables
EEG_EPOCHS=100 EEG_DIM=264 sbatch training/run_train_multigpu.sbatch

# Smaller batch for memory-constrained runs
EEG_BATCH_SIZE=16 EEG_DIM=264 sbatch training/run_train_multigpu.sbatch

# Single GPU run
EEG_NUM_GPUS=1 sbatch training/run_train_multigpu.sbatch
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EEG_DATA_DIR` | `datasets/model_data` | Path to processed data |
| `EEG_NUM_GPUS` | `4` | Number of GPUs (auto-capped to available) |
| `EEG_BATCH_SIZE` | `32` | Batch size per GPU |
| `EEG_EPOCHS` | `200` | Number of training epochs |
| `EEG_LR` | `0.0003` | Learning rate |
| `EEG_DIM` | `64` | Model dimension (64, 128, or 264) |
| `EEG_NUM_FOLDS` | `5` | Number of cross-validation folds (subject-aware) |
| `WANDB_PROJECT` | `eeg_spatial_spectral` | Weights & Biases project name |

#### How Multi-GPU Works

- Uses `torch.nn.DataParallel` for multi-GPU training
- Effective batch size = `EEG_BATCH_SIZE * EEG_NUM_GPUS`
- Gradient accumulation auto-adjusts to maintain ~32 effective batch size
- Model state_dict is saved unwrapped (without DataParallel prefix) for portability
- Data is staged to node-local SSD (`$SLURM_TMPDIR`) for faster I/O

#### Subject-Aware Cross-Validation (GroupKFold)

The training uses `sklearn.model_selection.GroupKFold` to ensure **no subject appears in both train and validation** within any fold. This prevents data leakage that inflates validation metrics.

**Previous approach (StratifiedKFold):** Split chunks randomly — chunks from the same subject could appear in both train and validation, making validation accuracy artificially high (essentially within-subject evaluation).

**Current approach (GroupKFold):** Groups chunks by subject ID (extracted from file names like `sub-077_eeg_chunk_5.set`). All chunks from a subject stay together in either train or validation.

| Setting | 5 folds (default) | 10 folds |
|---------|-------------------|----------|
| Train subjects/fold | ~56 | ~63 |
| Validation subjects/fold | ~14 | ~7 |
| Test subjects (cross) | 18 (untouched) | 18 (untouched) |

Per-fold logs show exactly which subjects are in validation:
```
Fold 1/5
  Train: 56 subjects (1750 chunks)
  Valid: 14 subjects (437 chunks) - [sub-001, sub-005, sub-012, ...]
```

#### What the SBATCH Script Does

1. Sets up Singularity container with `--nv` (GPU passthrough)
2. Stages `model_data/` to local SSD for fast I/O
3. Sets `PYTHONPATH` to include `datasets/`, `training/`, `layers/`, `utils/`, `data_provider/`
4. Prevents BLAS/OMP oversubscription (`OMP_NUM_THREADS=1`)
5. Configures WandB logging
6. Runs the training script inside the container

#### PYTHONPATH Setup

The training scripts use bare imports (e.g., `from eeg_dataset_multiscalegraph_spectral_advanced import ...`), so PYTHONPATH must include:

```
datasets/:training/:layers/:utils/:data_provider/:repo_root/
```

This is handled automatically by the sbatch script.

#### Monitoring

```bash
# Watch training output
tail -f /scratch/sd5963/slurm_logs/eeg_train_<JOBID>.out

# Watch errors
tail -f /scratch/sd5963/slurm_logs/eeg_train_<JOBID>.err

# Check job status
squeue -u sd5963

# GPU utilization (on compute node)
srun --jobid=<JOBID> nvidia-smi
```

#### WandB Integration

Training logs to Weights & Biases:
- Project: `eeg_spatial_spectral`
- Metrics logged: `train_loss`, `valid_loss`, `accuracy`, `f1_score`, `learning_rate`, `grad_norm`
- Config logged: all hyperparameters + `num_gpus`, `effective_batch_size`

---

## Training Guide

### Training the Best Model (Multiscale Graph Spectral)

#### On SLURM (recommended):
```bash
sbatch training/run_train_multigpu.sbatch
```

#### Locally with Singularity:
```bash
cd /scratch/sd5963/Neuroinformatics
singularity exec --nv /scratch/sd5963/containers/w2v2_cuda_cu128.sif \
    env PYTHONPATH="datasets:training:layers:utils:data_provider:." \
        EEG_DATA_DIR="datasets/model_data" \
        python3 training/train_kfold_multi_scale_graph_spectral_advanced.py
```

### Training Configuration

Key hyperparameters (set via env vars or edit script):

```python
num_channels = 19
num_classes = 3
dim = 64            # Model dimension: 64 (fast), 128, 264 (best accuracy)
scales = [3, 4, 5]  # Multi-scale factors
epochs = 200
learning_rate = 0.0003
batch_size = 32
weight_decay = 0.001
max_grad_norm = 0.3
time_length = 2000
```

### Training Pipeline Details

1. **Subject-aware Group K-Fold** cross-validation (5 folds default, configurable via `EEG_NUM_FOLDS`)
2. **Class balancing**: Downsamples majority classes to match minority
3. **Optimizer**: AdamW with warmup (5 epochs) + ReduceLROnPlateau
4. **Loss**: CrossEntropy (label smoothing=0.1) + auxiliary spatial/graph losses + contrastive loss
5. **Mixed precision**: `torch.amp.autocast` + `GradScaler`
6. **Early stopping**: Patience of 30 epochs on validation loss
7. **Checkpoints**: Every 5 epochs + best model per fold + best overall

### Training Other Models

```bash
# Multi-View Transformer
python3 training/train_kfold_mvt.py

# MVT with Wavelet features
python3 training/train_kfold_mvt_mfeat.py

# ADFormer
python3 training/train_kfold_adformer.py

# EEGNet
python3 training/train_kfold.py

# SVM with Grid Search
python3 training/train_kfold_svm_grid.py
```

### Resume Training from Checkpoint

Checkpoints are saved in `spatial_spectral_<timestamp>/checkpoints/`. Training automatically detects and resumes from the latest checkpoint.

---

## Inference & Testing

### Test the Best Model

```bash
cd /scratch/sd5963/Neuroinformatics/testing
singularity exec --nv /scratch/sd5963/containers/w2v2_cuda_cu128.sif \
    env PYTHONPATH="../datasets:../training:../layers:../utils:." \
        python3 test_multiscale_graph_spectral_advanced.py
```

### Test Other Models

```bash
python3 testing/test_mvt.py
python3 testing/test_adformer.py
python3 testing/test_svm.py
```

---

## Performance Benchmarks

### Model Comparison Table

| Model | Params | Within-Subject | Cross-Subject | Notes |
|-------|--------|----------------|---------------|-------|
| **Multiscale GS (264)** | 12.3M | **98.55%** | 59.68% | Best within-subject |
| Multiscale GS (128) | 5.8M | 97.38% | 63.46% | Lighter version |
| MVT | 8.2M | 89.83% | **64.95%** | Best cross-subject |
| MVT + Wavelet | 9.1M | 90.69% | 63.80% | |
| ADFormer | 6.5M | 85.20% | 58.30% | |
| EEGNet | 2.1M | 82.15% | 55.60% | |
| SVM | N/A | 78.50% | 52.30% | Baseline |

### Binary Classification Performance (Multiscale GS)

| Task | Accuracy |
|------|----------|
| AD vs CN | 98.53% |
| FTD vs CN | 98.99% |
| AD vs FTD | 98.17% |

---

## Known Issues

### 1. Mendeley Download URL is Dead

The original Mendeley dataset URL (`pbc979p2jc`) returns 404. Use `download_openneuro.py` (S3/boto3) instead. The `download_dataset.py` script is deprecated.

### 2. Data Directory Naming

`data_prep.py` creates `model_data/` (underscore). The training script previously referenced `model-data` (dash). This has been fixed - training now reads from `EEG_DATA_DIR` env var, defaulting to `model_data`.

### 3. Test Script Import Names Differ

The test script (`test_multiscale_graph_spectral_advanced.py`) imports from `eeg_multi_spatial_graph_spectral_advanced` (note: "spatial" not "scale"), which differs from the training script's `eeg_multi_scale_graph_spectral_advanced`. These module names need to be reconciled.

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
EEG_BATCH_SIZE=8 sbatch training/run_train_multigpu.sbatch

# Or reduce model dimension
EEG_DIM=64 sbatch training/run_train_multigpu.sbatch
```

### eeglabio Missing (Data Prep Only)

```bash
# Use --writable-tmpfs to install at runtime
singularity exec --writable-tmpfs /scratch/sd5963/containers/w2v2_cuda_cu128.sif \
    bash -c "pip install eeglabio -q && python3 data_prep.py"
```

### Import Errors (ModuleNotFoundError)

Ensure PYTHONPATH includes all module directories:

```bash
export PYTHONPATH="datasets:training:layers:utils:data_provider:."
```

The sbatch script handles this automatically.

### Training Instability

```python
# Lower learning rate
EEG_LR=0.0001

# The script already uses:
# - Gradient clipping (max_norm=0.3)
# - Warmup scheduler (5 epochs)
# - ReduceLROnPlateau (patience=10, factor=0.5)
# - Label smoothing (0.1)
```

### SLURM Job Debugging

```bash
# Check job output
cat /scratch/sd5963/slurm_logs/eeg_train_<JOBID>.out

# Check errors
cat /scratch/sd5963/slurm_logs/eeg_train_<JOBID>.err

# Interactive debug (single GPU)
srun --partition=l40s_public --gres=gpu:1 --mem=64G --cpus-per-task=8 \
    --account=torch_pr_60_tandon_advanced --time=1:00:00 --pty bash
```

---

## Changelog

### 2026-02-06: Subject-Aware Cross-Validation (GroupKFold)

**Problem:** Previous `StratifiedKFold` split at chunk level — chunks from the same subject could appear in both train and validation, causing data leakage and inflated validation accuracy.

**Solution:** Replaced with `sklearn.model_selection.GroupKFold` where groups = subject IDs.

**Changes to `train_kfold_multi_scale_graph_spectral_advanced.py`:**
- `StratifiedKFold` → `GroupKFold` (import + usage)
- Added `extract_subject_id()` to parse `sub-XXX` from chunk file names
- Builds `groups` array mapping each chunk to its subject after dataset creation
- Fold loop uses `gkf.split(..., groups=groups)` — all chunks from a subject stay together
- Per-fold logging shows which subjects are in validation and chunk counts
- New env var `EEG_NUM_FOLDS` (default: 5, configurable)

**Changes to `run_train_multigpu.sbatch`:**
- Added `EEG_NUM_FOLDS` env var, passed to singularity container

**Result:** With 5 folds and 70 training subjects: ~56 train / ~14 validation subjects per fold. 18 cross-subject test subjects remain completely untouched.

---

### 2026-02-06: Multi-GPU Training & Data Pipeline

**Dataset download:**
- Downloaded OpenNeuro ds004504 via S3 (boto3, unsigned access)
- 88 subjects, 4.5 GB total, all `.set` files verified
- Created `download_openneuro.py` as replacement for dead Mendeley URL

**Data preparation:**
- Ran `data_prep.py` inside Singularity container (with eeglabio)
- 4,436 total chunks: 3,219 train + 873 cross-test + 344 within-test
- Output in `datasets/model_data/`

**Multi-GPU training setup:**
- Created `training/run_train_multigpu.sbatch` (4x L40S, 128GB, 12h)
- Updated `train_kfold_multi_scale_graph_spectral_advanced.py`:
  - Added `nn.DataParallel` multi-GPU support
  - Added env var configuration (EEG_DATA_DIR, EEG_BATCH_SIZE, EEG_EPOCHS, EEG_LR, EEG_DIM, EEG_NUM_GPUS)
  - Fixed data path from `model-data` to `model_data`
  - Proper DataParallel unwrapping for model save/load/feature extraction
  - PYTHONPATH setup for bare module imports

---

*Last Updated: February 2026*
