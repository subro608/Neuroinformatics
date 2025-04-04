import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
import warnings
from datetime import datetime
import mne
import pywt

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# Import the CWT model from the file where you saved it
from cwt_eeg_model import create_cwt_model, CWTPreprocessor

# Ignore RuntimeWarning and FutureWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def load_eeg_data(file_path):
    """Load EEG data from EEGLAB file"""
    raw = mne.io.read_raw_eeglab(file_path)
    return raw.get_data()

class CWTEEGDataset(Dataset):
    """Dataset for EEG data with CWT preprocessing."""
    def __init__(self, data_directory, dataset_info, 
                 wavelet='morl', num_scales=32, 
                 min_freq=1, max_freq=30, sampling_rate=250):
        self.data_directory = data_directory
        self.dataset_info = dataset_info
        self.labels = [d['label'] for d in dataset_info]
        self.data = [d['file_name'] for d in dataset_info]
        
        # Initialize CWT preprocessor
        self.cwt_processor = CWTPreprocessor(
            wavelet=wavelet,
            num_scales=num_scales,
            min_freq=min_freq,
            max_freq=max_freq,
            sampling_rate=sampling_rate
        )
    
    def __len__(self):
        return len(self.dataset_info)
    
    def __getitem__(self, idx):
        file_info = self.dataset_info[idx]
        file_path = os.path.join(self.data_directory, file_info['file_name'])
        
        # Load raw EEG data
        eeg_data = load_eeg_data(file_path)
        
        # Apply CWT transform
        cwt_data = self.cwt_processor.transform(eeg_data)
        
        # Convert to tensor
        cwt_tensor = torch.FloatTensor(cwt_data)
        
        # Label (0 for 'A', 1 for 'C', 2 for 'F')
        label = 0 if file_info['label'] == 'A' else 1 if file_info['label'] == 'C' else 2
        
        return cwt_tensor, label

# Progress bar function
def progress_bar(current, total, width=50, prefix='', suffix=''):
    """Simple progress bar with percentage and visual indicator"""
    percent = f"{100 * (current / total):.1f}%"
    filled_length = int(width * current // total)
    bar = '=' * filled_length + '-' * (width - filled_length)
    print(f'\r{prefix} [{bar}] {percent} {suffix}', end='')
    # Print a newline when complete
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

set_seed(42)

def main():
    # Model and training parameters
    num_channels = 19
    num_classes = 3
    dim = 256
    num_scales = 32  # Number of CWT scales
    
    # CWT parameters
    wavelet = 'morl'  # Morlet wavelet
    min_freq = 1      # Minimum frequency (Hz)
    max_freq = 30     # Maximum frequency (Hz)
    sampling_rate = 250  # Sampling rate of EEG data
    
    # Model type and training parameters
    model_type = 'transformer'  # 'transformer' or 'cnn_transformer'
    batch_size = 16
    epochs = 100
    learning_rate = 0.0001  # Lower learning rate for transformer
    weight_decay = 0.0005
    accumulation_steps = 4
    max_grad_norm = 1.0  # Increased for more stable training
    num_layers = 3       # Number of transformer layers
    
    # Create directories for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'cwt_{model_type}_{timestamp}'
    
    for dir_path in [f'{base_dir}/images', f'{base_dir}/models', f'{base_dir}/logs', 
                     f'{base_dir}/checkpoints', f'{base_dir}/cwt_spectrograms']:
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
    model = create_cwt_model(
        model_type=model_type,
        num_channels=num_channels, 
        num_classes=num_classes, 
        dim=dim,
        num_scales=num_scales,
        num_layers=num_layers
    )
    
    # Log model architecture
    log_message(str(model))
    log_message(f"Model parameters: num_channels={num_channels}, num_classes={num_classes}, "
                f"dim={dim}, model_type={model_type}, num_scales={num_scales}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Total trainable parameters: {total_params:,}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Using device: {device}")
    model = model.to(device)
    
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
    
    # Create dataset with CWT features
    log_message("Creating dataset with CWT features...")
    train_dataset = CWTEEGDataset(
        data_dir, 
        balanced_train_data, 
        wavelet=wavelet, 
        num_scales=num_scales, 
        min_freq=min_freq, 
        max_freq=max_freq,
        sampling_rate=sampling_rate
    )
    
    # Debug: print statistics about the first batch to check data quality
    sample_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    sample_cwt, sample_labels = next(iter(sample_loader))
    
    log_message("\nCWT Data Statistics:")
    log_message(f"CWT Shape: {sample_cwt.shape}")
    log_message(f"Min: {sample_cwt.min().item():.4f}, Max: {sample_cwt.max().item():.4f}")
    log_message(f"Mean: {sample_cwt.mean().item():.4f}, Std: {sample_cwt.std().item():.4f}")
    
    # Save a sample CWT spectrogram for visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(sample_cwt[0, 0].numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Power')
    plt.title(f'CWT Spectrogram - Channel 0, Class: {sample_labels[0].item()}')
    plt.xlabel('Time')
    plt.ylabel('Scale (Frequency)')
    plt.savefig(f'{base_dir}/cwt_spectrograms/sample_cwt_spectrogram.png')
    plt.close()
    
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
        
        # Reset model for each fold
        model = create_cwt_model(
            model_type=model_type,
            num_channels=num_channels, 
            num_classes=num_classes, 
            dim=dim,
            num_scales=num_scales,
            num_layers=num_layers
        )
        model = model.to(device)
        
        # Prepare data loaders
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            pin_memory=True,
            num_workers=4
        )
        valid_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=valid_sampler,
            pin_memory=True,
            num_workers=4
        )
        
        log_message(f'Train dataloader: {len(train_dataloader)} batches')
        log_message(f'Valid dataloader: {len(valid_dataloader)} batches')
        
        # Initialize optimizer - Use AdamW for transformer
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
        
        # Initialize gradient scaler for mixed precision
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
            for batch_idx, (cwt_data, labels) in enumerate(train_dataloader):
                # Show batch progress
                progress_bar(batch_idx + 1, len(train_dataloader), width=40, 
                             prefix=f'Epoch {epoch + 1}/{epochs}, Batch:', 
                             suffix=f'{batch_idx + 1}/{len(train_dataloader)}')
                
                # Move data to device
                cwt_data = cwt_data.to(device)
                labels = labels.to(device)
                
                # Forward pass with mixed precision
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(cwt_data)
                    loss = criterion(outputs, labels) / accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
            
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
                for val_idx, (cwt_data, labels) in enumerate(valid_dataloader):
                    # Show validation progress
                    progress_bar(val_idx + 1, len(valid_dataloader), width=40, 
                                 prefix=f'Validating:', 
                                 suffix=f'{val_idx + 1}/{len(valid_dataloader)}')
                    
                    # Move data to device
                    cwt_data = cwt_data.to(device)
                    labels = labels.to(device)
                    
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(cwt_data)
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
            if accuracy > best_fold_accuracy:
                best_fold_accuracy = accuracy
                torch.save(model.state_dict(), f'{base_dir}/models/best_model_fold{fold+1}.pth')
                log_message(f'* New best model saved for fold {fold+1} with accuracy: {accuracy:.2f}% *')
                
                # If this is the best overall, update global best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), f'{base_dir}/models/best_model_overall.pth')
                    log_message(f'** New overall best model saved with accuracy: {accuracy:.2f}% **')
                    
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
                    plt.savefig(f'{base_dir}/images/confusion_matrix_best.png')
                    plt.close()
        
        # Store fold results
        train_losses.extend(fold_losses)
        valid_losses.extend(fold_valid_losses)
        all_fold_accuracies.append(best_fold_accuracy)
        all_fold_f1_scores.append(max(fold_f1_scores))
        
        # Plot fold results
        log_message(f"Plotting fold {fold+1} results...")
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(fold_losses, label='Train Loss')
        plt.plot(fold_valid_losses, label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - Fold {fold+1}')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(fold_accuracies, label='Accuracy')
        plt.plot(fold_f1_scores, label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(f'Accuracy and F1 Score - Fold {fold+1}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{base_dir}/images/metrics_fold{fold+1}.png')
        plt.close()
        
        # Detailed classification report
        class_names = ['A', 'C', 'F']
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        log_message(f"\nClassification Report - Fold {fold+1}:\n{report}")
    
    # Final results
    log_message('\nTraining completed!')
    log_message(f'Average Training Loss: {np.mean(train_losses):.4f}')
    log_message(f'Average Validation Loss: {np.mean(valid_losses):.4f}')
    log_message(f'Best Overall Accuracy: {best_accuracy:.2f}%')
    
    # Cross-validation results
    log_message('\nCross-Validation Results:')
    for i, (acc, f1) in enumerate(zip(all_fold_accuracies, all_fold_f1_scores)):
        log_message(f'Fold {i+1}: Accuracy = {acc:.2f}%, F1 Score = {f1:.4f}')
    
    log_message(f'Mean Accuracy: {np.mean(all_fold_accuracies):.2f}% ± {np.std(all_fold_accuracies):.2f}%')
    log_message(f'Mean F1 Score: {np.mean(all_fold_f1_scores):.4f} ± {np.std(all_fold_f1_scores):.4f}')
    
    # Plot overall results
    log_message("Plotting overall results...")
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(all_fold_accuracies)+1), all_fold_accuracies, alpha=0.7, label='Accuracy')
    plt.bar(range(1, len(all_fold_f1_scores)+1), all_fold_f1_scores, alpha=0.5, label='F1 Score')
    plt.axhline(y=np.mean(all_fold_accuracies), color='r', linestyle='-', label=f'Mean Acc: {np.mean(all_fold_accuracies):.2f}%')
    plt.axhline(y=np.mean(all_fold_f1_scores), color='g', linestyle='-', label=f'Mean F1: {np.mean(all_fold_f1_scores):.4f}')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Cross-Validation Results')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(all_fold_accuracies)+1))
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f'{base_dir}/images/cross_validation_results.png')
    plt.close()
    
    log_message('Training completed. Models and plots saved.')


if __name__ == "__main__":
    print("About to call main function")
    main()