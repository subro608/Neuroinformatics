import json
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import updated model and dataset
from eeg_mvt_dataset_mfeat import EEGDataset, create_data_loaders
from eeg_mvt_mfeat import MVTEEG
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

# Ignore RuntimeWarning and FutureWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable CUDA
mne.utils.set_config("MNE_USE_CUDA", "true")
mne.cuda.init_cuda(verbose=True)


# Progress bar function
def progress_bar(current, total, width=50, prefix="", suffix=""):
    """Simple progress bar with percentage and visual indicator"""
    percent = f"{100 * (current / total):.1f}%"
    filled_length = int(width * current // total)
    bar = "=" * filled_length + "-" * (width - filled_length)
    print(f"\r{prefix} [{bar}] {percent} {suffix}", end="")
    # Print a newline when complete
    if current == total:
        print()


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None

    checkpoint_info = []
    for ckpt in checkpoints:
        try:
            fold = int(ckpt.split("fold")[1].split("_")[0])
            epoch = int(ckpt.split("epoch")[1].split(".")[0])
            checkpoint_info.append((fold, epoch, ckpt))
        except:
            continue

    if not checkpoint_info:
        return None

    checkpoint_info.sort(key=lambda x: (x[0], x[1]))
    return checkpoint_info[-1]


def compute_metrics(y_true, y_pred):
    """Compute precision, recall, and F1 score"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return precision, recall, f1


# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# Create necessary directories
for dir_path in [
    "images_mvt_kfold_wavelet",
    "models_wavelet",
    "logs",
    "models_wavelet/checkpoints",
]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Create log file with timestamp
log_filename = f'logs/training_log_wavelet_{time.strftime("%Y%m%d_%H%M%S")}.txt'


def log_message(message, filename=log_filename):
    print(message)
    with open(filename, "a") as f:
        f.write(message + "\n")


# Model parameters
num_chans = 19
num_classes = 3
dim = 512
dropout_rate = 0.1


def init_weights(m):
    """Fixed initialization function using relu instead of gelu"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Initialize model with wavelet support
mvt_model = MVTEEG(num_channels=num_chans, num_classes=num_classes, dim=dim)
mvt_model.apply(init_weights)

# Log model architecture
log_message(str(mvt_model))
log_message(
    f"Model parameters: num_channels={num_chans}, num_classes={num_classes}, dim={dim}"
)
log_message("Model includes wavelet features")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mvt_model = mvt_model.to(device)

# Load and prepare data
data_dir = "model-data"
with open(os.path.join(data_dir, "labels.json"), "r") as file:
    data_info = json.load(file)

# Filter training data
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

log_message(f"Dataset Statistics:")
log_message(
    f"Before Balancing - A: {len(train_data_A)}, C: {len(train_data_C)}, F: {len(train_data_F)}"
)
log_message(f"After Balancing - A: {min_samples}, C: {min_samples}, F: {min_samples}")
log_message(f"Total samples: {len(balanced_train_data)}")

# Create dataset with wavelet features enabled
# Add progress tracking for dataset creation
log_message("Creating dataset with wavelet features enabled...")
log_message(
    "This may take some time as wavelet transforms are computed for each sample."
)


# Track dataset creation progress
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.start_time = time.time()

    def update(self, idx, filename):
        self.current += 1
        elapsed = time.time() - self.start_time
        samples_per_sec = self.current / elapsed if elapsed > 0 else 0
        remaining = (
            (self.total - self.current) / samples_per_sec if samples_per_sec > 0 else 0
        )

        progress_bar(
            self.current,
            self.total,
            width=50,
            prefix=f"Processing files:",
            suffix=f"{self.current}/{self.total} - {samples_per_sec:.2f} files/sec - {remaining:.1f}s left",
        )


# Create dataset with progress tracking
progress_tracker = ProgressTracker(len(balanced_train_data))


# Define progress callback function
def progress_callback(idx, filename):
    progress_tracker.update(idx, filename)


# Modify the EEGDataset to accept a progress callback
class ProgressEEGDataset(EEGDataset):
    def __init__(
        self,
        data_directory,
        dataset,
        normalize=True,
        include_wavelet=True,
        progress_callback=None,
    ):
        self.progress_callback = progress_callback
        super().__init__(data_directory, dataset, normalize, include_wavelet)

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        if self.progress_callback:
            self.progress_callback(idx, self.dataset[idx]["file_name"])
        return result


# Create dataset with progress tracking
train_dataset = ProgressEEGDataset(
    data_dir,
    balanced_train_data,
    include_wavelet=True,
    progress_callback=progress_callback,
)

# Training parameters
epochs = 100
batch_size = 16
learning_rate = 0.0003
accumulation_steps = 4
max_grad_norm = 1.0

# Resume training setup
checkpoint_dir = "models_wavelet/checkpoints"
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
start_fold = 0
start_epoch = 0
train_losses = []
valid_losses = []
best_accuracy = 0.0

if latest_checkpoint:
    fold_num, epoch_num, checkpoint_file = latest_checkpoint
    log_message(f"Resuming from checkpoint: Fold {fold_num}, Epoch {epoch_num}")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    checkpoint = torch.load(checkpoint_path)
    start_fold = checkpoint["fold"]
    start_epoch = checkpoint["epoch"] + 1
    train_losses = checkpoint.get("train_losses", [])
    valid_losses = checkpoint.get("valid_losses", [])
    best_accuracy = checkpoint.get("best_accuracy", 0.0)
else:
    log_message("Starting training from scratch")

# Initialize k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Training components
criterion = nn.CrossEntropyLoss()

# Training loop
for fold, (train_index, valid_index) in enumerate(
    skf.split(train_dataset.data, train_dataset.labels)
):
    if fold < start_fold:
        continue

    log_message(f"\nFold {fold + 1}/5")

    # Reset model for each fold
    mvt_model.apply(init_weights)

    # Prepare data loaders
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    valid_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    log_message(f"Train dataloader: {len(train_dataloader)} batches")
    log_message(f"Valid dataloader: {len(valid_dataloader)} batches")

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        mvt_model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95)
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.2,
        div_factor=10,
        final_div_factor=1e3,
    )

    # Load checkpoint if resuming
    if fold == start_fold and start_epoch > 0:
        mvt_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Initialize gradient scaler
    scaler = torch.amp.GradScaler()

    # Training epochs
    fold_losses = []
    fold_valid_losses = []

    for epoch in range(epochs):
        if fold == start_fold and epoch < start_epoch:
            continue

        epoch_start_time = time.time()
        mvt_model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        # Training loop
        for batch_idx, (inputs_dict, labels) in enumerate(train_dataloader):
            # Show batch progress
            progress_bar(
                batch_idx + 1,
                len(train_dataloader),
                width=40,
                prefix=f"Epoch {epoch + 1}/{epochs}, Batch:",
                suffix=f"{batch_idx + 1}/{len(train_dataloader)}",
            )

            # Move data to device
            inputs = {k: v.to(device) for k, v in inputs_dict.items()}
            labels = labels.to(device)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=device.type):
                # Pass wavelet features to the model
                outputs = mvt_model(inputs["time"])
                loss = criterion(outputs, labels) / accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    mvt_model.parameters(), max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * accumulation_steps

        # Validation
        mvt_model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # Validation loop with progress bar
        log_message("\nRunning validation...")
        with torch.no_grad():
            for val_idx, (inputs_dict, labels) in enumerate(valid_dataloader):
                # Show validation progress
                progress_bar(
                    val_idx + 1,
                    len(valid_dataloader),
                    width=40,
                    prefix=f"Validating:",
                    suffix=f"{val_idx + 1}/{len(valid_dataloader)}",
                )

                inputs = {k: v.to(device) for k, v in inputs_dict.items()}
                labels = labels.to(device)

                with torch.amp.autocast(device_type=device.type):
                    outputs = mvt_model(inputs["time"])
                    loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        epoch_train_loss = train_loss / len(train_dataloader)
        epoch_valid_loss = valid_loss / len(valid_dataloader)
        accuracy = 100.0 * correct / total
        precision, recall, f1 = compute_metrics(all_labels, all_preds)

        # Store losses
        fold_losses.append(epoch_train_loss)
        fold_valid_losses.append(epoch_valid_loss)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Log metrics
        metrics_message = (
            f"Fold {fold + 1}/5, Epoch {epoch + 1}/{epochs}\n"
            f"Time: {epoch_time:.2f}s\n"
            f"Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}\n"
            f"Accuracy: {accuracy:.2f}%, F1: {f1:.4f}\n"
            f"Precision: {precision:.4f}, Recall: {recall:.4f}\n"
            f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
        )
        log_message(metrics_message)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "fold": fold,
            "model_state_dict": mvt_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": epoch_train_loss,
            "valid_loss": epoch_valid_loss,
            "accuracy": accuracy,
            "best_accuracy": best_accuracy,
            "train_losses": train_losses + fold_losses,
            "valid_losses": valid_losses + fold_valid_losses,
        }

        torch.save(
            checkpoint, f"models_wavelet/checkpoints/fold{fold+1}_epoch{epoch+1}.pt"
        )

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                mvt_model.state_dict(),
                f"models_wavelet/best_mvt_wavelet_model_fold{fold+1}.pth",
            )
            log_message(f"★ New best model saved with accuracy: {accuracy:.2f}% ★")

    # Store fold results
    train_losses.extend(fold_losses)
    valid_losses.extend(fold_valid_losses)

    # Plot fold results
    log_message(f"Plotting fold {fold+1} results...")
    plt.figure(figsize=(12, 6))
    plt.plot(fold_losses, label="Train Loss")
    plt.plot(fold_valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss - Fold {fold+1}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images_mvt_kfold_wavelet/losses_fold{fold+1}.png")
    plt.close()

# Final results
log_message("\nTraining completed!")
log_message(f"Average Training Loss: {np.mean(train_losses):.4f}")
log_message(f"Average Validation Loss: {np.mean(valid_losses):.4f}")
log_message(f"Best Accuracy: {best_accuracy:.2f}%")

# Plot overall results
log_message("Plotting overall results...")
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overall Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("images_mvt_kfold_wavelet/losses_overall.png")
plt.close()

# Final comparison with baseline model (if available)
try:
    baseline_accuracy_file = "logs/best_baseline_accuracy.txt"
    if os.path.exists(baseline_accuracy_file):
        with open(baseline_accuracy_file, "r") as f:
            baseline_accuracy = float(f.read().strip())

        improvement = best_accuracy - baseline_accuracy
        log_message(f"Baseline model accuracy: {baseline_accuracy:.2f}%")
        log_message(f"Wavelet model improvement: {improvement:.2f}%")
except Exception as e:
    log_message(f"Could not compare with baseline: {e}")

log_message("Training completed. Models and plots saved.")
