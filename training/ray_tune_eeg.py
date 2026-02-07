#!/usr/bin/env python3
"""
Ray Tune hyperparameter search for EEG classification.

Supports both MVT and Multiscale Graph Spectral models.
Uses ASHA scheduler + Optuna (TPE) for efficient Bayesian HP optimization.
Single-fold evaluation per trial for speed; best config should be validated
with full 5-fold CV using the existing training scripts.

Usage:
    python3 training/ray_tune_eeg.py --num_samples 50 --max_epochs 100
"""

import argparse
import json
import os
import random
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import ray.train

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def compute_class_weights(labels, num_classes=3):
    """Compute inverse-frequency class weights for weighted loss."""
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 1.0
    return weights


# ---------------------------------------------------------------------------
# Trial function
# ---------------------------------------------------------------------------
def eeg_trial(config):
    """Run one hyperparameter configuration for Ray Tune."""

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb inside the trial
    wandb_run = None
    wandb_project = os.environ.get("WANDB_PROJECT", "eeg_ray_tune")
    if WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=wandb_project,
                config={k: v for k, v in config.items()
                        if k not in ("data_dir",)},
                reinit=True,
                group="ray_tune_eeg",
            )
        except Exception as e:
            print(f"[wandb] init failed (continuing without): {e}")
            wandb_run = None

    # ------------------------------------------------------------------
    # Unpack config
    # ------------------------------------------------------------------
    model_type = config["model_type"]
    dim = config["dim"]
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    batch_size = config["batch_size"]
    dropout = config["dropout"]
    label_smoothing = config["label_smoothing"]
    use_weighted_loss = config["use_weighted_loss"]
    augment_prob = config["augment_prob"]
    noise_std = config["noise_std"]
    max_grad_norm = config["max_grad_norm"]
    warmup_epochs = config["warmup_epochs"]
    patience = config["patience"]
    num_heads = config["num_heads"]
    max_epochs = config["max_epochs"]
    data_dir = config["data_dir"]
    fold_index = config["fold"]

    # ------------------------------------------------------------------
    # Load data and build split
    # ------------------------------------------------------------------
    with open(os.path.join(data_dir, "labels.json"), "r") as f:
        data_info = json.load(f)
    train_data = [d for d in data_info if d["type"] == "train"]

    # Balance classes
    by_label = {}
    for d in train_data:
        by_label.setdefault(d["label"], []).append(d)
    min_samples = min(len(v) for v in by_label.values())
    balanced_data = []
    for label_key in sorted(by_label.keys()):
        balanced_data.extend(random.sample(by_label[label_key], min_samples))

    # Subject-aware GroupKFold
    subject_ids = [extract_subject_id(d["file_name"]) for d in balanced_data]
    unique_subjects = sorted(set(subject_ids))
    subject_to_group = {s: i for i, s in enumerate(unique_subjects)}
    groups = np.array([subject_to_group[s] for s in subject_ids])

    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(range(len(balanced_data)), groups=groups))
    if fold_index >= len(splits):
        fold_index = 0
    train_index, valid_index = splits[fold_index]

    # ------------------------------------------------------------------
    # Build dataset and loaders
    # ------------------------------------------------------------------
    if model_type == "mvt":
        from eeg_mvt_dataset import EEGDataset

        dataset = EEGDataset(data_dir, balanced_data)
    else:  # multiscale_gs
        from eeg_dataset_multiscalegraph_spectral_advanced import (
            SpatialSpectralEEGDataset,
        )

        dataset = SpatialSpectralEEGDataset(
            data_dir=data_dir,
            data_info=balanced_data,
            scales=[3, 4, 5],
            time_length=2000,
            adjacency_type="combined",
        )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_index),
        pin_memory=pin_memory,
        num_workers=2,
        prefetch_factor=2,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(valid_index),
        pin_memory=pin_memory,
        num_workers=2,
        prefetch_factor=2,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Build augmentor
    # ------------------------------------------------------------------
    augmentor = None
    if augment_prob > 0:
        from eeg_augmentations import EEGAugmentor

        augmentor = EEGAugmentor(augment_prob=augment_prob, noise_std=noise_std)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    num_channels = 19
    num_classes = 3

    if model_type == "mvt":
        from eeg_mvt import create_model as create_mvt_model

        model = create_mvt_model(
            num_channels=num_channels, num_classes=num_classes, dim=dim
        )
    else:  # multiscale_gs
        from eeg_multi_scale_graph_spectral_advanced import (
            create_model as create_gs_model,
        )

        model = create_gs_model(
            num_channels=num_channels,
            num_classes=num_classes,
            dim=dim,
            scales=[3, 4, 5],
        )

    model.to(device)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    if use_weighted_loss:
        # Collect labels from the training fold
        train_labels = [dataset.labels[i] for i in train_index]
        class_weights = compute_class_weights(train_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ------------------------------------------------------------------
    # Optimizer & schedulers
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
    )
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: min(1.0, (e + 1) / warmup_epochs) if e < warmup_epochs else 1.0,
    )
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    scaler = torch.amp.GradScaler()
    best_val_loss = float("inf")
    patience_counter = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (inputs_data, labels) in enumerate(train_loader):
            labels = labels.to(device, non_blocking=True)

            # ----- Apply augmentations (training only) -----
            if model_type == "mvt":
                raw_eeg = inputs_data["time"].to(device, non_blocking=True)
                if augmentor is not None:
                    raw_eeg = augmentor(raw_eeg)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(raw_eeg)
                    loss = criterion(outputs, labels)
            else:
                raw_eeg = inputs_data["raw_eeg"].to(device, non_blocking=True)
                if augmentor is not None:
                    raw_eeg = augmentor(raw_eeg)

                scale_inputs = {
                    k: v.to(device, non_blocking=True)
                    for k, v in inputs_data.items()
                    if k.startswith("scale_")
                }
                if "adjacency" in inputs_data:
                    scale_inputs["adjacency"] = inputs_data["adjacency"].to(
                        device, non_blocking=True
                    )
                if "spatial_positions" in inputs_data:
                    scale_inputs["spatial_positions"] = inputs_data[
                        "spatial_positions"
                    ].to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(raw_eeg, scale_inputs)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()

        warmup_scheduler.step()
        avg_train_loss = train_loss / max(len(train_loader), 1)

        # ----- Validation -----
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs_data, labels in valid_loader:
                labels = labels.to(device, non_blocking=True)

                if model_type == "mvt":
                    raw_eeg = inputs_data["time"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(raw_eeg)
                        loss = criterion(outputs, labels)
                else:
                    raw_eeg = inputs_data["raw_eeg"].to(device, non_blocking=True)
                    scale_inputs = {
                        k: v.to(device, non_blocking=True)
                        for k, v in inputs_data.items()
                        if k.startswith("scale_")
                    }
                    if "adjacency" in inputs_data:
                        scale_inputs["adjacency"] = inputs_data["adjacency"].to(
                            device, non_blocking=True
                        )
                    if "spatial_positions" in inputs_data:
                        scale_inputs["spatial_positions"] = inputs_data[
                            "spatial_positions"
                        ].to(device, non_blocking=True)

                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(raw_eeg, scale_inputs)
                        loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = valid_loss / max(len(valid_loader), 1)
        val_accuracy = 100.0 * correct / max(total, 1)
        val_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        plateau_scheduler.step(avg_val_loss)

        # Report to Ray Tune
        metrics = {
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss,
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        ray.train.report(metrics)

        if wandb_run is not None:
            try:
                wandb.log(metrics)
            except Exception:
                pass

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Cleanup
    if wandb_run is not None:
        try:
            wandb.finish(quiet=True)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ray Tune HP search for EEG classification")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of HP configs to try")
    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs per trial")
    parser.add_argument("--gpus_per_trial", type=float, default=1, help="GPUs per trial")
    parser.add_argument("--cpus_per_trial", type=int, default=4, help="CPUs per trial")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to model_data dir")
    parser.add_argument("--fold", type=int, default=0, help="Which CV fold to use for validation")
    args = parser.parse_args()

    # Data directory
    data_dir = args.data_dir or os.environ.get(
        "EEG_DATA_DIR", "datasets/model_data"
    )

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # ------------------------------------------------------------------
    # Search space (14 parameters)
    # ------------------------------------------------------------------
    search_space = {
        "model_type": tune.choice(["mvt", "multiscale_gs"]),
        "dim": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([8, 16, 32]),
        "dropout": tune.uniform(0.1, 0.3),
        "label_smoothing": tune.uniform(0.0, 0.2),
        "use_weighted_loss": tune.choice([True, False]),
        "augment_prob": tune.uniform(0.0, 0.5),
        "noise_std": tune.uniform(0.01, 0.1),
        "max_grad_norm": tune.choice([0.3, 1.0, 3.0]),
        "warmup_epochs": tune.choice([3, 5, 10]),
        "patience": tune.choice([15, 30, 50]),
        "num_heads": tune.choice([4, 8, 16]),
        # Fixed per run
        "max_epochs": args.max_epochs,
        "data_dir": data_dir,
        "fold": args.fold,
    }

    # ------------------------------------------------------------------
    # Scheduler: ASHA
    # ------------------------------------------------------------------
    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=10,
        reduction_factor=3,
        metric="val_accuracy",
        mode="max",
    )

    # ------------------------------------------------------------------
    # Search algorithm: Optuna (TPE Bayesian)
    # ------------------------------------------------------------------
    search_alg = OptunaSearch(metric="val_accuracy", mode="max")

    # ------------------------------------------------------------------
    # Run Tune
    # ------------------------------------------------------------------
    wandb_project = os.environ.get("WANDB_PROJECT", "eeg_ray_tune")

    print(f"\n{'='*60}")
    print(f"Ray Tune EEG HP Search")
    print(f"  Samples:        {args.num_samples}")
    print(f"  Max epochs:     {args.max_epochs}")
    print(f"  GPUs/trial:     {args.gpus_per_trial}")
    print(f"  CPUs/trial:     {args.cpus_per_trial}")
    print(f"  Data dir:       {data_dir}")
    print(f"  Fold:           {args.fold}")
    print(f"  WandB project:  {wandb_project}")
    print(f"  WandB logging:  {'in-trial (direct)' if WANDB_AVAILABLE else 'disabled'}")
    print(f"{'='*60}\n")

    tuner = tune.Tuner(
        tune.with_resources(
            eeg_trial,
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=ray.train.RunConfig(
            name="eeg_ray_tune",
            storage_path=os.environ.get(
                "RAY_RESULTS_DIR",
                os.path.join("/scratch", os.environ.get("USER", "sd5963"), "ray_results"),
            ),
        ),
    )

    results = tuner.fit()

    # ------------------------------------------------------------------
    # Print best result
    # ------------------------------------------------------------------
    best_result = results.get_best_result(metric="val_accuracy", mode="max")
    print(f"\n{'='*60}")
    print("BEST TRIAL")
    print(f"{'='*60}")
    print(f"  Val Accuracy: {best_result.metrics['val_accuracy']:.2f}%")
    print(f"  Val F1:       {best_result.metrics['val_f1']:.4f}")
    print(f"  Val Loss:     {best_result.metrics['val_loss']:.4f}")
    print(f"\n  Config:")
    for k, v in best_result.config.items():
        if k not in ("max_epochs", "data_dir", "fold"):
            print(f"    {k}: {v}")
    print(f"{'='*60}")

    # Save best config to file
    results_dir = os.environ.get(
        "RAY_RESULTS_DIR",
        os.path.join("/scratch", os.environ.get("USER", "sd5963"), "ray_results"),
    )
    os.makedirs(results_dir, exist_ok=True)
    best_config_path = os.path.join(results_dir, "best_eeg_config.json")
    with open(best_config_path, "w") as f:
        serializable = {
            k: v
            for k, v in best_result.config.items()
            if isinstance(v, (int, float, str, bool))
        }
        serializable["best_val_accuracy"] = best_result.metrics["val_accuracy"]
        serializable["best_val_f1"] = best_result.metrics["val_f1"]
        json.dump(serializable, f, indent=2)
    print(f"\nBest config saved to: {best_config_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
