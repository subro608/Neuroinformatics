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
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# Import our custom models and dataset
from eeg_multi_spatial_graph_spectral import create_model, MVTSpatialSpectralModel
from eeg_dataset_multispatialgraph_spectral import SpatialSpectralEEGDataset

# Ignore RuntimeWarning and FutureWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Progress bar function
def progress_bar(current, total, width=50, prefix='', suffix=''):
    """Simple progress bar with percentage and visual indicator"""
    percent = f"{100 * (current / float(total)):.1f}%"
    filled_length = int(width * current // total)
    bar = '=' * filled_length + '-' * (width - filled_length)
    print(f'\r{prefix} [{bar}] {percent} {suffix}', end='')
    if current == total:
        print()

def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
    
    checkpoint_info = []
    for ckpt in checkpoints:
        try:
            fold = int(ckpt.split('fold')[1].split('_')[0])
            epoch = int(ckpt.split('epoch')[1].split('.')[0])
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
        y_true, y_pred, 
        average='weighted',
        zero_division=0
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

def main():
    # Set random seed
    set_seed(42)
    
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Model and training parameters
    num_channels = 19
    num_classes = 3
    dim = 64  # Increased from 32 to 64 for better expressiveness
    scales = [3, 4, 5]  # Multi-scale approach
    
    batch_size = 16  # Increased from 2 to better utilize GPU
    epochs = 200
    learning_rate = 0.0003
    weight_decay = 0.001
    accumulation_steps = 1  # Reduced since we increased batch size
    max_grad_norm = 0.3
    time_length = 2000
    
    # Set up WandB
    run_name = f"spatial_spectral_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="eeg_spatial_spectral",
        name=run_name,
        config={
            "model_type": "MVTSpatialSpectral_Simplified",
            "dim": dim,
            "scales": scales,
            "num_channels": num_channels,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "accumulation_steps": accumulation_steps,
            "max_grad_norm": max_grad_norm,
            "optimizer": "AdamW",
            "time_length": time_length
        }
    )
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'spatial_spectral_{timestamp}'
    
    for dir_path in [f'{base_dir}/models', f'{base_dir}/logs', f'{base_dir}/checkpoints']:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Create log file
    log_filename = f'{base_dir}/logs/training_log.txt'
    
    def log_message(message, filename=log_filename):
        print(message)
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        except UnicodeEncodeError:
            # Fallback to ASCII with character replacement
            simple_message = message.replace('★', '*')
            with open(filename, 'a') as f:
                f.write(simple_message + '\n')
    
    # Initialize model
    model = create_model(
        num_channels=num_channels,
        num_classes=num_classes,
        dim=dim,
        scales=scales
    )
    
    log_message(f"Using simplified MVTSpatialSpectral model with scales {scales}")
    
    # Log model architecture
    log_message(str(model))
    log_message(f"Model parameters: num_channels={num_channels}, num_classes={num_classes}, "
                f"dim={dim}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Total trainable parameters: {total_params:,}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Using device: {device}")
    model = model.to(device)
    
    # Optimize CUDA performance
    if device.type == 'cuda':
        log_message(f"CUDA device: {torch.cuda.get_device_name(0)}")
        log_message(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Enable half-precision 
        model = model.half()
        log_message("Using half precision (FP16) for training")
    
    # Watch the model in wandb
    wandb.watch(model, log="all", log_freq=10)
    
    # Load and prepare data
    data_dir = 'model-data'
    with open(os.path.join(data_dir, 'labels.json'), 'r') as file:
        data_info = json.load(file)
    
    # Filter training data
    train_data = [d for d in data_info if d['type'] == 'train']
    
    # Balance dataset
    train_data_A = [d for d in train_data if d['label'] == 'A']
    train_data_C = [d for d in train_data if d['label'] == 'C']
    train_data_F = [d for d in train_data if d['label'] == 'F']
    
    min_samples = min(len(train_data_A), len(train_data_C), len(train_data_F))
    balanced_train_data = (
        random.sample(train_data_A, min_samples) +
        random.sample(train_data_C, min_samples) +
        random.sample(train_data_F, min_samples)
    )
    
    log_message(f'Dataset Statistics:')
    log_message(f'Before Balancing - A: {len(train_data_A)}, C: {len(train_data_C)}, F: {len(train_data_F)}')
    log_message(f'After Balancing - A: {min_samples}, C: {min_samples}, F: {min_samples}')
    log_message(f'Total samples: {len(balanced_train_data)}')
    
    # Log dataset info to wandb
    wandb.log({
        "dataset_size_before_balancing": len(train_data),
        "dataset_size_after_balancing": len(balanced_train_data),
        "class_A_samples": min_samples,
        "class_C_samples": min_samples,
        "class_F_samples": min_samples
    })
    
    # Create enhanced dataset with spatial features and data augmentation
    log_message("Creating enhanced spatial-spectral dataset...")
    train_dataset = SpatialSpectralEEGDataset(
        data_dir=data_dir, 
        data_info=balanced_train_data, 
        scales=scales,
        time_length=time_length,
        augment=True,
        adjacency_type='combined'
    )
    
    # Debug: print stats about the first batch to check data quality
    sample_batch, sample_labels = next(iter(DataLoader(train_dataset, batch_size=4)))
    
    log_message("\nData Statistics:")
    data_stats = {}
    
    # Check raw EEG data
    if 'raw_eeg' in sample_batch:
        raw_eeg = sample_batch['raw_eeg']
        log_message(f"Raw EEG - Shape: {raw_eeg.shape}, "
                f"Min: {raw_eeg.min().item():.4f}, Max: {raw_eeg.max().item():.4f}, "
                f"Mean: {raw_eeg.mean().item():.4f}, Std: {raw_eeg.std().item():.4f}")
        data_stats["raw_eeg_min"] = raw_eeg.min().item()
        data_stats["raw_eeg_max"] = raw_eeg.max().item()
        data_stats["raw_eeg_mean"] = raw_eeg.mean().item()
        data_stats["raw_eeg_std"] = raw_eeg.std().item()
    
    # Check spectral embeddings
    for scale in scales:
        features = sample_batch[f'scale_{scale}']
        log_message(f"Scale {scale} - Shape: {features.shape}, "
                f"Min: {features.min().item():.4f}, Max: {features.max().item():.4f}, "
                f"Mean: {features.mean().item():.4f}, Std: {features.std().item():.4f}")
        data_stats[f"scale_{scale}_min"] = features.min().item()
        data_stats[f"scale_{scale}_max"] = features.max().item()
        data_stats[f"scale_{scale}_mean"] = features.mean().item()
        data_stats[f"scale_{scale}_std"] = features.std().item()
    
    # Check adjacency if available
    if 'adjacency' in sample_batch:
        log_message(f"Adjacency - Shape: {sample_batch['adjacency'].shape}, "
                f"Min: {sample_batch['adjacency'].min().item():.4f}, "
                f"Max: {sample_batch['adjacency'].max().item():.4f}")
        data_stats["adjacency_min"] = sample_batch['adjacency'].min().item()
        data_stats["adjacency_max"] = sample_batch['adjacency'].max().item()
    
    # Check spatial positions if available
    if 'spatial_positions' in sample_batch:
        log_message(f"Spatial Positions - Shape: {sample_batch['spatial_positions'].shape}, "
                f"Min: {sample_batch['spatial_positions'].min().item():.4f}, "
                f"Max: {sample_batch['spatial_positions'].max().item():.4f}")
        data_stats["spatial_positions_min"] = sample_batch['spatial_positions'].min().item()
        data_stats["spatial_positions_max"] = sample_batch['spatial_positions'].max().item()
    
    # Log data statistics to wandb
    wandb.log(data_stats)
    
    # Resume training setup
    checkpoint_dir = f'{base_dir}/checkpoints'
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
        start_fold = checkpoint['fold']
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        valid_losses = checkpoint.get('valid_losses', [])
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
    else:
        log_message("Starting training from scratch")
    
    # Initialize k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Training components
    # Use Label Smoothing CE Loss for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Track all fold results
    all_fold_accuracies = []
    all_fold_f1_scores = []
    
    # Training loop
    for fold, (train_index, valid_index) in enumerate(skf.split(train_dataset.data, train_dataset.labels)):
        if fold < start_fold:
            continue
            
        log_message(f'\nFold {fold + 1}/5')
        # Log fold to wandb
        wandb.log({"current_fold": fold + 1})
        
        # Reset model for each fold
        model = create_model(
            num_channels=num_channels,
            num_classes=num_classes,
            dim=dim,
            scales=scales
        )
        model = model.to(device)
        
        # Not using half-precision to avoid FP16 gradient issues
        if device.type == 'cuda':
            log_message("Using full precision (FP32) for training")
        
        # Prepare data loaders
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)
        
        # Use pin_memory for faster data transfer to GPU
        pin_memory = True if device.type == 'cuda' else False
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            pin_memory=pin_memory,
            num_workers=4,  # Increased worker count for better data loading
            prefetch_factor=2,
            drop_last=True
        )
        
        valid_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=valid_sampler,
            pin_memory=pin_memory,
            num_workers=4,
            prefetch_factor=2,
            drop_last=True
        )
        
        log_message(f'Train dataloader: {len(train_dataloader)} batches')
        log_message(f'Valid dataloader: {len(valid_dataloader)} batches')
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        def warmup_cosine_schedule(epoch):
            if epoch < 5:  # Warmup for 5 epochs
                return float(epoch) / 5.0
            else:
                # Cosine annealing after warmup
                return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (epochs - 5)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
        
        # Load checkpoint if resuming
        if fold == start_fold and start_epoch > 0:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Initialize gradient scaler for mixed precision training
        scaler = torch.amp.GradScaler()
        
        # Training epochs
        fold_losses = []
        fold_valid_losses = []
        fold_accuracies = []
        fold_f1_scores = []
        best_fold_accuracy = 0.0
        patience_counter = 0
        best_fold_loss = float('inf')
        
        for epoch in range(epochs):
            if fold == start_fold and epoch < start_epoch:
                continue
                
            epoch_start_time = time.time()
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            # For timing data loading vs. model execution
            data_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            
            # Training loop
            for batch_idx, (inputs_dict, labels) in enumerate(train_dataloader):
                # Show batch progress
                progress_bar(batch_idx + 1, len(train_dataloader), width=40, 
                             prefix=f'Epoch {epoch + 1}/{epochs}, Batch:', 
                             suffix=f'{batch_idx + 1}/{len(train_dataloader)}')
                
                data_start = time.time()
                
                # Move data to device with non-blocking transfer (using FP32)
                if device.type == 'cuda':
                    raw_eeg = inputs_dict['raw_eeg'].to(device, non_blocking=True)
                    scale_inputs = {k: v.to(device, non_blocking=True) 
                                  for k, v in inputs_dict.items() if k.startswith('scale_')}
                    
                    if 'adjacency' in inputs_dict:
                        scale_inputs['adjacency'] = inputs_dict['adjacency'].to(device, non_blocking=True)
                    
                    if 'spatial_positions' in inputs_dict:
                        scale_inputs['spatial_positions'] = inputs_dict['spatial_positions'].to(device, non_blocking=True)
                else:
                    raw_eeg = inputs_dict['raw_eeg'].to(device, non_blocking=True)
                    scale_inputs = {k: v.to(device, non_blocking=True) 
                                  for k, v in inputs_dict.items() if k.startswith('scale_')}
                    
                    if 'adjacency' in inputs_dict:
                        scale_inputs['adjacency'] = inputs_dict['adjacency'].to(device, non_blocking=True)
                    
                    if 'spatial_positions' in inputs_dict:
                        scale_inputs['spatial_positions'] = inputs_dict['spatial_positions'].to(device, non_blocking=True)
                
                labels = labels.to(device, non_blocking=True)
                
                data_time += time.time() - data_start
                forward_start = time.time()
                
                # Forward pass with mixed precision
                with torch.amp.autocast(device_type=device.type):
                    # Pass both raw EEG and embeddings_dict to the model
                    outputs = model(raw_eeg, scale_inputs)
                    loss = criterion(outputs, labels) / accumulation_steps
                
                forward_time += time.time() - forward_start
                backward_start = time.time()
                
                # Backward pass
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Log gradient norm to wandb
                    wandb.log({"grad_norm": grad_norm})
                
                backward_time += time.time() - backward_start
                train_loss += loss.item() * accumulation_steps
                
                # Log batch loss to wandb every 10 batches
                if batch_idx % 10 == 0:
                    wandb.log({
                        "batch_loss": loss.item() * accumulation_steps,
                        "epoch": epoch + 1,
                        "batch": batch_idx
                    })
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Validation
            model.eval()
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
                    progress_bar(val_idx + 1, len(valid_dataloader), width=40, 
                                 prefix=f'Validating:', 
                                 suffix=f'{val_idx + 1}/{len(valid_dataloader)}')
                    
                    # Move data to device (using FP32)
                    if device.type == 'cuda':
                        raw_eeg = inputs_dict['raw_eeg'].to(device, non_blocking=True)
                        scale_inputs = {k: v.to(device, non_blocking=True) 
                                      for k, v in inputs_dict.items() if k.startswith('scale_')}
                        
                        if 'adjacency' in inputs_dict:
                            scale_inputs['adjacency'] = inputs_dict['adjacency'].to(device, non_blocking=True)
                        
                        if 'spatial_positions' in inputs_dict:
                            scale_inputs['spatial_positions'] = inputs_dict['spatial_positions'].to(device, non_blocking=True)
                    else:
                        raw_eeg = inputs_dict['raw_eeg'].to(device, non_blocking=True)
                        scale_inputs = {k: v.to(device, non_blocking=True) 
                                      for k, v in inputs_dict.items() if k.startswith('scale_')}
                        
                        if 'adjacency' in inputs_dict:
                            scale_inputs['adjacency'] = inputs_dict['adjacency'].to(device, non_blocking=True)
                        
                        if 'spatial_positions' in inputs_dict:
                            scale_inputs['spatial_positions'] = inputs_dict['spatial_positions'].to(device, non_blocking=True)
                    
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
            
            # Calculate metrics
            epoch_train_loss = train_loss / len(train_dataloader)
            epoch_valid_loss = valid_loss / len(valid_dataloader)
            accuracy = 100. * correct / total
            precision, recall, f1 = compute_metrics(all_labels, all_preds)
            
            # Store metrics
            fold_losses.append(epoch_train_loss)
            fold_valid_losses.append(epoch_valid_loss)
            fold_accuracies.append(accuracy)
            fold_f1_scores.append(f1)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Track memory usage
            if device.type == 'cuda':
                max_memory = torch.cuda.max_memory_allocated() / 1024**2
                current_memory = torch.cuda.memory_allocated() / 1024**2
                wandb.log({
                    "max_gpu_memory": max_memory,
                    "current_gpu_memory": current_memory
                })
                log_message(f"Max GPU memory used: {max_memory:.2f} MB")
            
            # Log timing breakdown
            log_message(f"Time breakdown - Data loading: {data_time:.2f}s, Forward: {forward_time:.2f}s, Backward: {backward_time:.2f}s")
            
            # Log metrics
            metrics_message = (
                f'Fold {fold + 1}/5, Epoch {epoch + 1}/{epochs}\n'
                f'Time: {epoch_time:.2f}s\n'
                f'Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}\n'
                f'Accuracy: {accuracy:.2f}%, F1: {f1:.4f}\n'
                f'Precision: {precision:.4f}, Recall: {recall:.4f}\n'
                f'Learning Rate: {current_lr:.6f}'
            )
            log_message(metrics_message)
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "fold": fold + 1,
                "train_loss": epoch_train_loss,
                "valid_loss": epoch_valid_loss,
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
                "data_loading_time": data_time,
                "forward_time": forward_time,
                "backward_time": backward_time
            })
            
            # Save checkpoint every 5 epochs or on the last epoch
            if epoch % 5 == 0 or epoch == epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'fold': fold,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': epoch_train_loss,
                    'valid_loss': epoch_valid_loss,
                    'accuracy': accuracy,
                    'best_accuracy': best_accuracy,
                    'train_losses': train_losses + fold_losses,
                    'valid_losses': valid_losses + fold_valid_losses
                }
                torch.save(checkpoint, f'{checkpoint_dir}/fold{fold+1}_epoch{epoch+1}.pt')
            
            # Early stopping logic
            if epoch_valid_loss < best_fold_loss:
                best_fold_loss = epoch_valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Save best model based on accuracy
            if accuracy > best_fold_accuracy:
                best_fold_accuracy = accuracy
                torch.save(model.state_dict(), f'{base_dir}/models/best_model_fold{fold+1}.pth')
                log_message(f'* New best model saved for fold {fold+1} with accuracy: {accuracy:.2f}% *')
                
                # If this is the best overall, update global best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), f'{base_dir}/models/best_model_overall.pth')
                    log_message(f'** New overall best model saved with accuracy: {accuracy:.2f}% **')
            
            # Apply early stopping if no improvement for 30 epochs
            if patience_counter >= 30:
                log_message(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss.")
                break
        
        # End of fold
        # Store fold results
        fold_avg_accuracy = sum(fold_accuracies[-10:]) / min(10, len(fold_accuracies))
        fold_avg_f1 = sum(fold_f1_scores[-10:]) / min(10, len(fold_f1_scores))
        all_fold_accuracies.append(fold_avg_accuracy)
        all_fold_f1_scores.append(fold_avg_f1)
        
        log_message(f'\nFold {fold+1} completed.')
        log_message(f'Average accuracy over last 10 epochs: {fold_avg_accuracy:.2f}%')
        log_message(f'Average F1 score over last 10 epochs: {fold_avg_f1:.4f}')
        log_message(f'Best accuracy: {best_fold_accuracy:.2f}%')
        
        # Log fold summary to wandb
        wandb.log({
            f"fold_{fold+1}_avg_accuracy": fold_avg_accuracy,
            f"fold_{fold+1}_avg_f1": fold_avg_f1,
            f"fold_{fold+1}_best_accuracy": best_fold_accuracy
        })
        
        # Free up memory between folds
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # End of training
    # Compute overall performance
    if all_fold_accuracies:
        overall_accuracy = sum(all_fold_accuracies) / len(all_fold_accuracies)
        overall_f1 = sum(all_fold_f1_scores) / len(all_fold_f1_scores)
        
        log_message('\n' + '='*50)
        log_message(f'Training completed!')
        log_message(f'Average accuracy across all folds: {overall_accuracy:.2f}%')
        log_message(f'Average F1 score across all folds: {overall_f1:.4f}')
        log_message(f'Best accuracy achieved: {best_accuracy:.2f}%')
        
        # Log final summary to wandb
        wandb.log({
            "overall_accuracy": overall_accuracy,
            "overall_f1_score": overall_f1,
            "best_accuracy": best_accuracy
        })
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()