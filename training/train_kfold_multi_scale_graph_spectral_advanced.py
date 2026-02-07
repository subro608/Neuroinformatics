import json
import os
import random
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from eeg_dataset_multiscalegraph_spectral_advanced import SpatialSpectralEEGDataset

# Import custom models and dataset
from eeg_multi_scale_graph_spectral_advanced import (
    MVTSpatialSpectralModel,
    create_model,
)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, SubsetRandomSampler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def progress_bar(current, total, width=50, prefix="", suffix=""):
    percent = f"{100 * (current / float(total)):.1f}%"
    filled_length = int(width * current // total)
    bar = "=" * filled_length + "-" * (width - filled_length)
    print(f"\r{prefix} [{bar}] {percent} {suffix}", end="")
    if current == total:
        print()


def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None
    checkpoint_info = [
        (
            int(f.split("fold")[1].split("_")[0]),
            int(f.split("epoch")[1].split(".")[0]),
            f,
        )
        for f in checkpoints
        if "fold" in f and "epoch" in f
    ]
    return max(checkpoint_info, key=lambda x: (x[0], x[1]), default=None)


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return precision, recall, f1


def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_subject_id(file_name):
    """Extract subject ID (e.g., 'sub-077') from chunk file name."""
    basename = os.path.basename(file_name)
    return basename.split("_eeg_chunk")[0]


def main():
    set_seed(42)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Model and training parameters (overridable via env vars)
    num_channels = 19
    num_classes = 3
    dim = int(os.environ.get("EEG_DIM", 64))
    scales = [3, 4, 5]
    epochs = int(os.environ.get("EEG_EPOCHS", 200))
    learning_rate = float(os.environ.get("EEG_LR", 0.0003))
    weight_decay = 0.001
    max_grad_norm = 0.3
    time_length = 2000

    # Multi-GPU setup
    num_gpus = int(os.environ.get("EEG_NUM_GPUS", 1))
    if torch.cuda.is_available():
        num_gpus = min(num_gpus, torch.cuda.device_count())
    else:
        num_gpus = 0
    use_multi_gpu = num_gpus > 1

    # Batch size: configurable via env var, scaled by number of GPUs
    if os.environ.get("EEG_BATCH_SIZE"):
        batch_size = int(os.environ["EEG_BATCH_SIZE"])
    elif torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        batch_size = (
            32 if free_mem > 8e9 else 16 if free_mem > 4e9 else 8
        )
    else:
        batch_size = 4
    # With DataParallel, effective batch = batch_size * num_gpus
    effective_batch = batch_size * max(num_gpus, 1)
    accumulation_steps = max(1, 32 // effective_batch)

    # WandB setup
    run_name = f"spatial_spectral_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "eeg_spatial_spectral"),
        name=run_name,
        config={
            "model_type": "MVTSpatialSpectral_Enhanced",
            "dim": dim,
            "scales": scales,
            "num_channels": num_channels,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "effective_batch_size": effective_batch,
            "num_gpus": num_gpus,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "accumulation_steps": accumulation_steps,
            "max_grad_norm": max_grad_norm,
            "optimizer": "AdamW",
            "time_length": time_length,
        },
    )
    print(f"Multi-GPU: {use_multi_gpu} ({num_gpus} GPUs, effective batch: {effective_batch})")

    # Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"spatial_spectral_{timestamp}"
    for dir_path in [
        f"{base_dir}/models",
        f"{base_dir}/logs",
        f"{base_dir}/checkpoints",
    ]:
        os.makedirs(dir_path, exist_ok=True)

    log_filename = f"{base_dir}/logs/training_log.txt"

    def log_message(message):
        print(message)
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    # Initialize model
    model = create_model(
        num_channels=num_channels, num_classes=num_classes, dim=dim, scales=scales
    )
    log_message(f"Using enhanced MVTSpatialSpectral model with scales {scales}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Total trainable parameters: {total_params:,}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if use_multi_gpu:
        model = nn.DataParallel(model)
        log_message(f"Using DataParallel across {num_gpus} GPUs")
    log_message(f"Using device: {device}")
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            log_message(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    wandb.watch(model, log="all", log_freq=10)

    # Load data
    data_dir = os.environ.get("EEG_DATA_DIR", "model_data")
    with open(os.path.join(data_dir, "labels.json"), "r") as file:
        data_info = json.load(file)
    train_data = [d for d in data_info if d["type"] == "train"]

    # Balance dataset
    train_data_A = [d for d in train_data if d["label"] == "A"]
    train_data_C = [d for d in train_data if d["label"] == "C"]
    train_data_F = [d for d in train_data if d["label"] == "F"]
    min_samples = min(len(train_data_A), len(train_data_C), len(train_data_F))
    balanced_train_data = (
        random.sample(train_data_A, min_samples)
        + random.sample(train_data_C, min_samples)
        + random.sample(train_data_F, min_samples)
    )

    log_message(
        f"Dataset Statistics: Before - A: {len(train_data_A)}, C: {len(train_data_C)}, F: {len(train_data_F)}"
    )
    log_message(
        f"After - A: {min_samples}, C: {min_samples}, F: {min_samples}, Total: {len(balanced_train_data)}"
    )
    wandb.log(
        {
            "dataset_size_after_balancing": len(balanced_train_data),
            "class_samples": min_samples,
        }
    )

    # Create dataset
    train_dataset = SpatialSpectralEEGDataset(
        data_dir=data_dir,
        data_info=balanced_train_data,
        scales=scales,
        time_length=time_length,
        adjacency_type="combined",
    )

    # Build subject groups for subject-aware cross-validation
    subject_ids = [extract_subject_id(d["file_name"]) for d in balanced_train_data]
    unique_subjects = sorted(set(subject_ids))
    subject_to_group = {s: i for i, s in enumerate(unique_subjects)}
    groups = np.array([subject_to_group[s] for s in subject_ids])
    log_message(f"Subject-aware CV: {len(unique_subjects)} unique subjects, {len(subject_ids)} chunks")

    # Debug batch
    sample_loader = DataLoader(train_dataset, batch_size=4)
    sample_batch, sample_labels = next(iter(sample_loader))
    log_message("\nData Statistics:")
    for key in sample_batch:
        log_message(
            f"{key} - Shape: {sample_batch[key].shape}, Min: {sample_batch[key].min():.4f}, Max: {sample_batch[key].max():.4f}"
        )
    wandb.log({f"{key}_stats": sample_batch[key].shape[1:] for key in sample_batch})

    # Resume training
    checkpoint_dir = f"{base_dir}/checkpoints"
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    start_fold, start_epoch = 0, 0
    best_accuracy = 0.0
    if latest_checkpoint:
        fold_num, epoch_num, checkpoint_file = latest_checkpoint
        checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file))
        start_fold, start_epoch = checkpoint["fold"], checkpoint["epoch"] + 1
        best_accuracy = checkpoint.get("best_accuracy", 0.0)
        log_message(f"Resuming from Fold {fold_num}, Epoch {epoch_num}")
    else:
        log_message("Starting from scratch")

    # Subject-aware Group K-Fold (no data leakage between subjects)
    n_folds = int(os.environ.get("EEG_NUM_FOLDS", 5))
    gkf = GroupKFold(n_splits=n_folds)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    contrastive_criterion = nn.CosineEmbeddingLoss(margin=0.5)

    for fold, (train_index, valid_index) in enumerate(
        gkf.split(train_dataset.data, train_dataset.labels, groups=groups)
    ):
        if fold < start_fold:
            continue

        val_subjects = sorted(set(subject_ids[i] for i in valid_index))
        train_subjects_fold = sorted(set(subject_ids[i] for i in train_index))
        log_message(f"\nFold {fold + 1}/{n_folds}")
        log_message(f"  Train: {len(train_subjects_fold)} subjects ({len(train_index)} chunks)")
        log_message(f"  Valid: {len(val_subjects)} subjects ({len(valid_index)} chunks) - {val_subjects}")
        wandb.log({"current_fold": fold + 1, "val_subjects": len(val_subjects), "train_subjects": len(train_subjects_fold)})

        # Reset model for each fold
        model = create_model(
            num_channels=num_channels, num_classes=num_classes, dim=dim, scales=scales
        ).to(device)
        if use_multi_gpu:
            model = nn.DataParallel(model)

        # Data loaders
        pin_memory = device.type == "cuda"
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_index),
            pin_memory=pin_memory,
            num_workers=4,
            prefetch_factor=2,
            drop_last=True,
        )
        valid_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(valid_index),
            pin_memory=pin_memory,
            num_workers=4,
            prefetch_factor=2,
            drop_last=True,
        )

        log_message(
            f"Train batches: {len(train_dataloader)}, Valid batches: {len(valid_dataloader)}"
        )

        # Optimizer and schedulers
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: min(1.0, e / 5.0) if e < 5 else 1.0
        )
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        if fold == start_fold and start_epoch > 0:
            model_to_load = model.module if use_multi_gpu else model
            model_to_load.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scaler = torch.amp.GradScaler()
        best_fold_accuracy, best_fold_loss = 0.0, float("inf")
        patience_counter = 0

        # Auxiliary classifiers
        aux_spatial = nn.Linear(dim, num_classes).to(device)
        aux_graph = nn.Linear(dim, num_classes).to(device)

        for epoch in range(start_epoch if fold == start_fold else 0, epochs):
            model.train()
            train_loss, data_time, forward_time, backward_time = 0.0, 0.0, 0.0, 0.0
            optimizer.zero_grad()

            for batch_idx, (inputs_dict, labels) in enumerate(train_dataloader):
                progress_bar(
                    batch_idx + 1,
                    len(train_dataloader),
                    prefix=f"Epoch {epoch + 1}/{epochs}, Batch:",
                )
                data_start = time.time()

                raw_eeg = inputs_dict["raw_eeg"].to(device, non_blocking=True)
                scale_inputs = {
                    k: v.to(device, non_blocking=True)
                    for k, v in inputs_dict.items()
                    if k.startswith("scale_")
                }
                if "adjacency" in inputs_dict:
                    scale_inputs["adjacency"] = inputs_dict["adjacency"].to(
                        device, non_blocking=True
                    )
                if "spatial_positions" in inputs_dict:
                    scale_inputs["spatial_positions"] = inputs_dict[
                        "spatial_positions"
                    ].to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                data_time += time.time() - data_start
                forward_start = time.time()

                # Forward pass - FIXED: Remove checkpointing that was causing issues
                with torch.amp.autocast(device_type=device.type):
                    # Direct forward pass without checkpointing
                    outputs = model(raw_eeg, scale_inputs)

                    # Extract intermediate features (assuming model exposes them)
                    base_model = model.module if use_multi_gpu else model
                    spatial_pooled = base_model.spatial_extractor(
                        raw_eeg, scale_inputs.get("spatial_positions", None)
                    ).mean(0)
                    graph_pooled = base_model.graph_module(scale_inputs)

                    if graph_pooled.size(1) != dim:
                        graph_pooled = F.pad(
                            graph_pooled, (0, dim - graph_pooled.size(1))
                        )

                    loss_ce = criterion(outputs, labels) / accumulation_steps
                    loss_aux_spatial = (
                        criterion(aux_spatial(spatial_pooled), labels)
                        / accumulation_steps
                    )
                    loss_aux_graph = (
                        criterion(aux_graph(graph_pooled), labels) / accumulation_steps
                    )
                    loss_contrastive = (
                        contrastive_criterion(
                            spatial_pooled,
                            graph_pooled,
                            torch.ones(batch_size).to(device),
                        )
                        / accumulation_steps
                    )
                    loss = (
                        loss_ce
                        + 0.3 * (loss_aux_spatial + loss_aux_graph)
                        + 0.1 * loss_contrastive
                    )

                forward_time += time.time() - forward_start
                backward_start = time.time()

                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    wandb.log({"grad_norm": grad_norm})

                backward_time += time.time() - backward_start
                train_loss += loss.item() * accumulation_steps

            warmup_scheduler.step()
            epoch_train_loss = train_loss / len(train_dataloader)

            # Validation
            model.eval()
            valid_loss, correct, total = 0.0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs_dict, labels in valid_dataloader:
                    raw_eeg = inputs_dict["raw_eeg"].to(device, non_blocking=True)
                    scale_inputs = {
                        k: v.to(device, non_blocking=True)
                        for k, v in inputs_dict.items()
                        if k.startswith("scale_")
                    }
                    if "adjacency" in inputs_dict:
                        scale_inputs["adjacency"] = inputs_dict["adjacency"].to(
                            device, non_blocking=True
                        )
                    if "spatial_positions" in inputs_dict:
                        scale_inputs["spatial_positions"] = inputs_dict[
                            "spatial_positions"
                        ].to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(raw_eeg, scale_inputs)
                        loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            epoch_valid_loss = valid_loss / len(valid_dataloader)
            accuracy = 100.0 * correct / total
            precision, recall, f1 = compute_metrics(all_labels, all_preds)

            plateau_scheduler.step(epoch_valid_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Logging
            epoch_time = time.time() - time.time()
            log_message(
                f"Fold {fold + 1}/{n_folds}, Epoch {epoch + 1}/{epochs}, "
                f"Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}, "
                f"Accuracy: {accuracy:.2f}%, F1: {f1:.4f}, LR: {current_lr:.6f}"
            )
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "fold": fold + 1,
                    "train_loss": epoch_train_loss,
                    "valid_loss": epoch_valid_loss,
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "learning_rate": current_lr,
                }
            )

            # Checkpoint (unwrap DataParallel for portable state_dict)
            model_to_save = model.module if use_multi_gpu else model
            if epoch % 5 == 0 or epoch == epochs - 1:
                torch.save(
                    {
                        "epoch": epoch,
                        "fold": fold,
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_accuracy": best_accuracy,
                    },
                    f"{checkpoint_dir}/fold{fold+1}_epoch{epoch+1}.pt",
                )

            # Early stopping and best model
            if accuracy > best_fold_accuracy:
                best_fold_accuracy = accuracy
                torch.save(
                    model_to_save.state_dict(), f"{base_dir}/models/best_model_fold{fold+1}.pth"
                )
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(
                        model_to_save.state_dict(), f"{base_dir}/models/best_model_overall.pth"
                    )

            if epoch_valid_loss < best_fold_loss:
                best_fold_loss = epoch_valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 30:
                    log_message(f"Early stopping at epoch {epoch + 1}")
                    break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final summary
    wandb.finish()


if __name__ == "__main__":
    main()
