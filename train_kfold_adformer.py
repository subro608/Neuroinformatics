import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
import warnings
import os
from datetime import datetime
import wandb
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import torch.nn.functional as F

# Import ADformer model and vanilla dataset
from eeg_adformer import create_adformer_eeg_model, count_parameters
from eeg_dataset_adformer import create_vanilla_adformer_dataset, create_vanilla_adformer_dataloader

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def progress_bar(current, total, width=50, prefix='', suffix=''):
    percent = f"{100 * (current / float(total)):.1f}%"
    filled_length = int(width * current // total)
    bar = '=' * filled_length + '-' * (width - filled_length)
    print(f'\r{prefix} [{bar}] {percent} {suffix}', end='')
    if current == total:
        print()

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
    checkpoint_info = [(int(f.split('fold')[1].split('_')[0]), int(f.split('epoch')[1].split('.')[0]), f) 
                       for f in checkpoints if 'fold' in f and 'epoch' in f]
    return max(checkpoint_info, key=lambda x: (x[0], x[1]), default=None)

def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return precision, recall, f1, acc, conf_matrix

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_dataset_stats(dataset, num_samples=100):
    """Calculate channel-wise mean and std from a random subset of samples"""
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    all_data = []
    
    for idx in indices:
        sample, _ = dataset[idx]
        eeg = sample['eeg'].numpy()  # Shape: [seq_len, channels]
        all_data.append(eeg)
    
    all_data = np.concatenate(all_data, axis=0)  # Shape: [num_samples*seq_len, channels]
    # Calculate stats along first dimension (time)
    channel_means = np.mean(all_data, axis=0)  # Shape: [channels]
    channel_stds = np.std(all_data, axis=0)  # Shape: [channels]
    
    # Replace near-zero stds to avoid division by zero
    channel_stds[channel_stds < 1e-6] = 1.0
    
    return channel_means, channel_stds

def log_data_stats(dataset, name, log_message_fn, num_samples=50):
    """Log statistics about the dataset samples"""
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    channel_means = []
    channel_stds = []
    for idx in indices:
        sample, _ = dataset[idx]
        eeg = sample['eeg'].numpy()  # Shape: [seq_len, channels]
        # Calculate stats along first dimension (time)
        channel_means.append(np.mean(eeg, axis=0))
        channel_stds.append(np.std(eeg, axis=0))
    
    overall_mean = np.mean(channel_means, axis=0)
    overall_std = np.mean(channel_stds, axis=0)
    
    # Log overall statistics
    log_message_fn(f"{name} dataset statistics:")
    log_message_fn(f"- Channel means: {np.round(overall_mean, 4)}")
    log_message_fn(f"- Channel stds: {np.round(overall_std, 4)}")
    
    # Log min/max values
    min_vals = []
    max_vals = []
    for idx in indices:
        sample, _ = dataset[idx]
        eeg = sample['eeg'].numpy()
        min_vals.append(np.min(eeg))
        max_vals.append(np.max(eeg))
    
    log_message_fn(f"- Min value: {np.min(min_vals)}, Max value: {np.max(max_vals)}")
    return overall_mean, overall_std

def plot_sample_eeg(dataset, idx, title, save_path):
    """Plot a sample EEG signal for visualization"""
    sample, label = dataset[idx]
    eeg = sample['eeg'].numpy()  # Shape: [seq_len, channels]
    
    # Plot first 5 channels
    plt.figure(figsize=(12, 8))
    for i in range(min(5, eeg.shape[1])):
        plt.subplot(5, 1, i+1)
        plt.plot(eeg[:, i])
        plt.title(f'Channel {i+1}')
        if i == 0:
            plt.title(f'{title} - Label: {label} - Channel {i+1}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    set_seed(42)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Model and training parameters
    num_channels = 19
    num_classes = 3
    d_model = 32  # Reduced from 32
    time_length = 2000
    
    # ADformer-specific parameters - SIMPLIFIED
    patch_len_list = "400"  # Single temporal scale
    up_dim_list = "15"      # Single channel scale
    augmentations = "none,jitter,scale"  # Reduced augmentations
    dropout = 0.3  # Increased dropout
    
    # Training parameters
    epochs = 100  # Reduced number of epochs
    learning_rate = 1e-4  # Increased learning rate
    weight_decay = 0.01  # Increased weight decay
    max_grad_norm = 0.5  # Reduced gradient norm
    l1_lambda = 5e-4  # Increased L1 regularization

    # Dynamic batch size
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        batch_size = 32 if free_mem > 8e9 else 16 if free_mem > 4e9 else 8
    else:
        batch_size = 4
    accumulation_steps = max(1, 32 // batch_size)

    # WandB setup
    run_name = f"simplified_adformer_eeg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="adformer_eeg", name=run_name, config={
        "model_type": "Simplified_ADformer_EEG",
        "d_model": d_model,
        "patch_len_list": patch_len_list,
        "up_dim_list": up_dim_list,
        "augmentations": augmentations,
        "num_channels": num_channels,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "accumulation_steps": accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "optimizer": "AdamW",
        "time_length": time_length,
        "dropout": dropout,
        "l1_lambda": l1_lambda
    })

    # Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'simplified_adformer_eeg_{timestamp}'
    for dir_path in [f'{base_dir}/models', f'{base_dir}/logs', f'{base_dir}/checkpoints', f'{base_dir}/plots']:
        os.makedirs(dir_path, exist_ok=True)

    log_filename = f'{base_dir}/logs/training_log.txt'
    def log_message(message):
        print(message)
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    # Load data
    data_dir = 'model-data'
    with open(os.path.join(data_dir, 'labels.json'), 'r') as file:
        data_info = json.load(file)
    train_data = [d for d in data_info if d['type'] == 'train']

    # Balance dataset
    train_data_A = [d for d in train_data if d['label'] == 'A']
    train_data_C = [d for d in train_data if d['label'] == 'C']
    train_data_F = [d for d in train_data if d['label'] == 'F']
    min_samples = min(len(train_data_A), len(train_data_C), len(train_data_F))
    balanced_train_data = (random.sample(train_data_A, min_samples) + 
                           random.sample(train_data_C, min_samples) + 
                           random.sample(train_data_F, min_samples))
    
    log_message(f'Dataset Statistics: Before - A: {len(train_data_A)}, C: {len(train_data_C)}, F: {len(train_data_F)}')
    log_message(f'After - A: {min_samples}, C: {min_samples}, F: {min_samples}, Total: {len(balanced_train_data)}')
    wandb.log({"dataset_size_after_balancing": len(balanced_train_data), "class_samples": min_samples})

    # K-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)  # Removed label smoothing

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Using device: {device}")
    if device.type == 'cuda':
        log_message(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Resume training
    checkpoint_dir = f'{base_dir}/checkpoints'
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    start_fold, start_epoch = 0, 0
    best_accuracy = 0.0
    if latest_checkpoint:
        fold_num, epoch_num, checkpoint_file = latest_checkpoint
        checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file))
        start_fold, start_epoch = checkpoint['fold'], checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        log_message(f"Resuming from Fold {fold_num}, Epoch {epoch_num}")
    else:
        log_message("Starting from scratch")

    # Get all labels for stratified K-fold
    all_labels = [0 if d['label'] == 'A' else 1 if d['label'] == 'C' else 2 for d in balanced_train_data]
    
    # Generate fold indices using stratified K-fold on balanced data
    fold_indices = list(skf.split(balanced_train_data, all_labels))

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        if fold < start_fold:
            continue

        log_message(f'\nFold {fold + 1}/5')
        wandb.log({"current_fold": fold + 1})

        # Create training and validation datasets directly with appropriate subsets
        train_data_info = [balanced_train_data[i] for i in train_idx]
        valid_data_info = [balanced_train_data[i] for i in val_idx]
        
        # Create datasets - note we're creating separate datasets now, not using samplers
        train_dataset = create_vanilla_adformer_dataset(
            data_dir=data_dir, 
            data_info=train_data_info,  # Only training data 
            time_length=time_length, 
            patch_lengths=[int(p) for p in patch_len_list.split(",")],
            up_dimensions=[int(d) for d in up_dim_list.split(",")],
            augmentations=augmentations.split(","),
            normalize=False,  # We'll handle normalization manually
            is_training=True
        )
        
        valid_dataset = create_vanilla_adformer_dataset(
            data_dir=data_dir, 
            data_info=valid_data_info,  # Only validation data
            time_length=time_length, 
            patch_lengths=[int(p) for p in patch_len_list.split(",")],
            up_dimensions=[int(d) for d in up_dim_list.split(",")],
            augmentations=["none"],  # No augmentation for validation
            normalize=False,  # We'll handle normalization manually
            is_training=False
        )
        
        # Calculate normalization stats properly from training data
        channel_means, channel_stds = calculate_dataset_stats(train_dataset)
        
        # Apply the same normalization manually to both datasets
        train_dataset.set_normalization_stats(channel_means, channel_stds)
        valid_dataset.set_normalization_stats(channel_means, channel_stds)
        
        # Log statistics about both datasets
        train_stats_mean, train_stats_std = log_data_stats(train_dataset, "Training", log_message)
        valid_stats_mean, valid_stats_std = log_data_stats(valid_dataset, "Validation", log_message)
        
        # Plot sample EEGs for visual inspection
        if len(train_dataset) > 0 and len(valid_dataset) > 0:
            plot_sample_eeg(train_dataset, 0, "Training Sample", f"{base_dir}/plots/train_sample_fold{fold+1}.png")
            plot_sample_eeg(valid_dataset, 0, "Validation Sample", f"{base_dir}/plots/valid_sample_fold{fold+1}.png")
        
        # Compare distributions
        log_message(f"Mean difference between train and valid: {np.mean(np.abs(train_stats_mean - valid_stats_std))}")
        log_message(f"Std difference between train and valid: {np.mean(np.abs(train_stats_std - valid_stats_std))}")

        # Log validation class distribution
        valid_labels = [valid_dataset.labels[i] for i in range(len(valid_dataset))]
        class_counts = {0: valid_labels.count(0), 1: valid_labels.count(1), 2: valid_labels.count(2)}
        log_message(f"Validation class distribution for fold {fold + 1}: A: {class_counts[0]}, C: {class_counts[1]}, F: {class_counts[2]}")

        # Initialize SIMPLIFIED model for this fold
        model = create_adformer_eeg_model(
            enc_in=num_channels,
            seq_len=time_length,
            patch_len_list=patch_len_list,  # Single temporal scale
            up_dim_list=up_dim_list,        # Single channel scale
            d_model=d_model,                # Smaller hidden dimension
            n_heads=6,                      # Single head
            e_layers=2,                     # Single layer
            d_ff=d_model*2,                 # Smaller feed-forward
            dropout=dropout,
            augmentations=augmentations,
            num_class=num_classes
        ).to(device)

        # Initialize weights using Xavier/Glorot
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        log_message(f"Using Simplified ADformer EEG model with temporal patches: {patch_len_list}, channel dimensions: {up_dim_list}")
        total_params = count_parameters(model)
        log_message(f"Total trainable parameters: {total_params:,}")
        wandb.watch(model, log="all", log_freq=10)

        # Data loaders - no samplers now, just shuffle the training data
        train_dataloader = create_vanilla_adformer_dataloader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # Shuffle now instead of using a sampler
            sampler=None,  # No sampler needed
            num_workers=0
        )
        
        valid_dataloader = create_vanilla_adformer_dataloader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )

        log_message(f'Train batches: {len(train_dataloader)}, Valid batches: {len(valid_dataloader)}')

        # Optimizer and scheduler - improved learning rate strategy
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # One-cycle LR with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_dataloader),
            pct_start=0.1,  # 10% warmup
            div_factor=25.0,  # initial_lr = max_lr/25
            final_div_factor=10000.0  # min_lr = initial_lr/10000
        )

        if fold == start_fold and start_epoch > 0:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scaler = torch.amp.GradScaler()
        best_fold_accuracy, best_fold_loss = 0.0, float('inf')
        patience_counter = 0
        epoch_valid_losses = []
        
        # Check model gradients and outputs on a sample batch before training
        if len(train_dataloader) > 0:
            sample_batch = next(iter(train_dataloader))
            adformer_inputs, _ = train_dataset.get_batch_for_adformer(sample_batch)
            x_enc = adformer_inputs['x_enc'].to(device, non_blocking=True)
            x_mark_enc = adformer_inputs['x_mark_enc'].to(device, non_blocking=True)
            adformer_labels = adformer_inputs['labels'].to(device, non_blocking=True)
            
            # Forward pass
            with torch.no_grad():
                sample_outputs = model(x_enc, x_mark_enc)
                log_message(f"Initial model outputs - min: {sample_outputs.min().item()}, max: {sample_outputs.max().item()}")
                log_message(f"Initial model outputs - mean: {sample_outputs.mean().item()}, std: {sample_outputs.std().item()}")
                initial_pred = sample_outputs.argmax(dim=1)
                log_message(f"Initial predictions distribution: {torch.bincount(initial_pred, minlength=num_classes)}")

        for epoch in range(start_epoch if fold == start_fold else 0, epochs):
            model.train()
            train_loss, data_time, forward_time, backward_time = 0.0, 0.0, 0.0, 0.0
            optimizer.zero_grad()
            
            all_train_preds = []
            all_train_labels = []

            for batch_idx, (inputs_dict, labels) in enumerate(train_dataloader):
                progress_bar(batch_idx + 1, len(train_dataloader), prefix=f'Epoch {epoch + 1}/{epochs}, Batch:')
                data_start = time.time()

                adformer_inputs, _ = train_dataset.get_batch_for_adformer((inputs_dict, labels))
                x_enc = adformer_inputs['x_enc'].to(device, non_blocking=True)
                x_mark_enc = adformer_inputs['x_mark_enc'].to(device, non_blocking=True)
                adformer_labels = adformer_inputs['labels'].to(device, non_blocking=True)

                data_time += time.time() - data_start
                forward_start = time.time()

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(x_enc, x_mark_enc)
                    ce_loss = criterion(outputs, adformer_labels) / accumulation_steps
                    
                    # Add L1 regularization
                    l1_loss = 0
                    for param in model.parameters():
                        l1_loss += torch.norm(param, 1)
                    loss = ce_loss + l1_lambda * l1_loss / accumulation_steps

                forward_time += time.time() - forward_start
                backward_start = time.time()

                scaler.scale(loss).backward()
                
                # Track predictions for training metrics
                _, predicted = outputs.max(1)
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(adformer_labels.cpu().numpy())
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Update the learning rate every batch with OneCycle
                    scheduler.step()
                    
                    if batch_idx % 10 == 0:  # Log every 10 batches
                        wandb.log({
                            "grad_norm": grad_norm,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "batch_loss": ce_loss.item() * accumulation_steps
                        })

                backward_time += time.time() - backward_start
                train_loss += ce_loss.item() * accumulation_steps  # Log only the CE loss

            epoch_train_loss = train_loss / len(train_dataloader)
            
            # Calculate training metrics
            train_precision, train_recall, train_f1, train_acc, _ = compute_metrics(all_train_labels, all_train_preds)
            
            # Validation
            model.eval()
            valid_loss, correct, total = 0.0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs_dict, labels in valid_dataloader:
                    adformer_inputs, _ = valid_dataset.get_batch_for_adformer((inputs_dict, labels))
                    x_enc = adformer_inputs['x_enc'].to(device, non_blocking=True)
                    x_mark_enc = adformer_inputs['x_mark_enc'].to(device, non_blocking=True)
                    adformer_labels = adformer_inputs['labels'].to(device, non_blocking=True)

                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(x_enc, x_mark_enc)
                        loss = criterion(outputs, adformer_labels)
                    
                    valid_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += adformer_labels.size(0)
                    correct += predicted.eq(adformer_labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(adformer_labels.cpu().numpy())

            epoch_valid_loss = valid_loss / len(valid_dataloader)
            epoch_valid_losses.append(epoch_valid_loss)
            accuracy = 100. * correct / total
            precision, recall, f1, acc, conf_matrix = compute_metrics(all_labels, all_preds)

            current_lr = optimizer.param_groups[0]['lr']
            log_message(f'Fold {fold + 1}/5, Epoch {epoch + 1}/{epochs}, '
                        f'Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}, '
                        f'Train Acc: {train_acc:.2f}%, Valid Acc: {accuracy:.2f}%, '
                        f'F1: {f1:.4f}, LR: {current_lr:.6f}')
            
            wandb.log({
                "epoch": epoch + 1, 
                "fold": fold + 1, 
                "train_loss": epoch_train_loss, 
                "valid_loss": epoch_valid_loss, 
                "train_accuracy": train_acc * 100,
                "valid_accuracy": accuracy, 
                "train_f1": train_f1,
                "valid_f1": f1, 
                "precision": precision,
                "recall": recall,
                "learning_rate": current_lr,
                "confusion_matrix": wandb.plot.confusion_matrix(
                    preds=np.array(all_preds), 
                    y_true=np.array(all_labels),
                    class_names=["A", "C", "F"]
                )
            })

            if epoch % 5 == 0 or epoch == epochs - 1:
                torch.save({
                    'epoch': epoch, 
                    'fold': fold, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'best_accuracy': best_accuracy
                }, f'{checkpoint_dir}/fold{fold+1}_epoch{epoch+1}.pt')

            if accuracy > best_fold_accuracy:
                best_fold_accuracy = accuracy
                torch.save(model.state_dict(), f'{base_dir}/models/best_model_fold{fold+1}.pth')
                log_message(f"New best model for fold {fold+1} with accuracy: {accuracy:.2f}%")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), f'{base_dir}/models/best_model_overall.pth')
                    log_message(f"New best overall model with accuracy: {accuracy:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check if validation loss is increasing for too many epochs
            if len(epoch_valid_losses) >= 5:
                if all(epoch_valid_losses[-1] > epoch_valid_losses[-i-1] for i in range(1, min(5, len(epoch_valid_losses)))):
                    log_message(f"Validation loss increasing for consecutive epochs. Early stopping.")
                    break
            
            if patience_counter >= 10:  # Early stopping based on accuracy
                log_message(f"No improvement in accuracy for 10 epochs. Early stopping.")
                break

        # Plot the learning curves for this fold
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epoch_valid_losses)
        plt.title(f'Fold {fold+1} Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'{base_dir}/plots/fold{fold+1}_valid_loss.png')
        plt.close()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    log_message(f"\nTraining completed. Best overall accuracy: {best_accuracy:.2f}%")
    wandb.log({"best_overall_accuracy": best_accuracy})
    wandb.finish()

if __name__ == "__main__":
    main()