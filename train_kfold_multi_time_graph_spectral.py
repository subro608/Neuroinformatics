import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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

# Import our custom models
from eeg_multi_time_graph_spectral import create_model, MVTTimeMultiScaleModel
from eeg_dataset_multitimegraph_spectral import MultiScaleTimeSpectralEEGDataset

# Ignore RuntimeWarning and FutureWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Progress bar function
def progress_bar(current, total, width=50, prefix='', suffix=''):
    """Simple progress bar with percentage and visual indicator"""
    percent = f"{100 * (current / total):.1f}%"
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
    
    # Model and training parameters
    num_channels = 19
    num_classes = 3
    dim = 256
    scales = [3, 4, 5]  # Multi-scale approach
    
    batch_size = 16
    epochs = 200
    learning_rate = 0.0001
    weight_decay = 0.0005
    accumulation_steps = 4
    max_grad_norm = 0.1
    
    # Initialize wandb with meaningful run name
    run_name = f"mvt_timegraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="eeg_mvt_timegraph",
        name=run_name,
        config={
            "model_type": "MVTTimeMultiScale",
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
            "optimizer": "AdamW"
        }
    )
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'mvt_timegraph_{timestamp}'
    
    for dir_path in [f'{base_dir}/images', f'{base_dir}/models', f'{base_dir}/logs', 
                     f'{base_dir}/checkpoints', f'{base_dir}/connectivity']:
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
    
    log_message(f"Using MVTTimeMultiScale model with scales {scales}")
    
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
    
    # Create dataset with multi-scale features
    log_message("Creating dataset with spectral features and raw EEG signals...")
    train_dataset = MultiScaleTimeSpectralEEGDataset(data_dir, balanced_train_data, scales=scales)
    
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
    criterion = nn.CrossEntropyLoss()
    
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
        
        # Prepare data loaders
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
        
        log_message(f'Train dataloader: {len(train_dataloader)} batches')
        log_message(f'Valid dataloader: {len(valid_dataloader)} batches')
        
        # Initialize optimizer - Use AdamW
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Use cosine annealing schedule with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2  # Double period after each restart
        )
        
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
        
        for epoch in range(epochs):
            if fold == start_fold and epoch < start_epoch:
                continue
                
            epoch_start_time = time.time()
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            # Training loop
            for batch_idx, (inputs_dict, labels) in enumerate(train_dataloader):
                # Show batch progress
                progress_bar(batch_idx + 1, len(train_dataloader), width=40, 
                             prefix=f'Epoch {epoch + 1}/{epochs}, Batch:', 
                             suffix=f'{batch_idx + 1}/{len(train_dataloader)}')
                
                # Move data to device
                # For our combined model, we need both raw EEG and spectral embeddings
                raw_eeg = inputs_dict['raw_eeg'].to(device)
                scale_inputs = {k: v.to(device) for k, v in inputs_dict.items() if k.startswith('scale_')}
                labels = labels.to(device)
                
                # Forward pass with mixed precision
                with torch.amp.autocast(device_type=device.type):
                    # Pass both raw EEG and embeddings_dict to the model
                    outputs = model(raw_eeg, scale_inputs)
                    loss = criterion(outputs, labels) / accumulation_steps
                
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
                    
                    # Move data to device
                    raw_eeg = inputs_dict['raw_eeg'].to(device)
                    scale_inputs = {k: v.to(device) for k, v in inputs_dict.items() if k.startswith('scale_')}
                    labels = labels.to(device)
                    
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
                "epoch_time": epoch_time
            })
            
            # Save checkpoint
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
            
            # Save best model
            # After saving best model based on accuracy
            if accuracy > best_fold_accuracy:
                best_fold_accuracy = accuracy
                torch.save(model.state_dict(), f'{base_dir}/models/best_model_fold{fold+1}.pth')
                log_message(f'* New best model saved for fold {fold+1} with accuracy: {accuracy:.2f}% *')
                
                # If this is the best overall, update global best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), f'{base_dir}/models/best_model_overall.pth')
                    log_message(f'** New overall best model saved with accuracy: {accuracy:.2f}% **')
                    
                    try:
                        # Create and save confusion matrix for best model
                        cm = confusion_matrix(all_labels, all_preds)
                        plt.figure(figsize=(10, 8))
                        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        plt.title(f'Confusion Matrix - Fold {fold+1} (Acc: {accuracy:.2f}%)')
                        plt.colorbar()
                        class_names = ['A', 'C', 'F']
                        tick_marks = np.arange(len(class_names))
                        plt.xticks(tick_marks, class_names)
                        plt.yticks(tick_marks, class_names)
                        
                        # Add text annotations to confusion matrix
                        thresh = cm.max() / 2.
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                plt.text(j, i, format(cm[i, j], 'd'),
                                        horizontalalignment="center",
                                        color="white" if cm[i, j] > thresh else "black")
                        
                        plt.ylabel('True label')
                        plt.xlabel('Predicted label')
                        plt.tight_layout()
                        
                        # Save to local file first
                        confusion_matrix_path = f'{base_dir}/images/confusion_matrix_best.png'
                        plt.savefig(confusion_matrix_path)
                        
                        # Log to wandb if file exists
                        if os.path.exists(confusion_matrix_path):
                            wandb.log({"confusion_matrix": wandb.Image(confusion_matrix_path)})
                        
                    except Exception as e:
                        log_message(f"Warning: Error creating or logging confusion matrix: {e}")
                    finally:
                        # Make sure to close the plot even if there's an error
                        plt.close()
        
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