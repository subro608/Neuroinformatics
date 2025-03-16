import os
import json
import mne
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

from eeg_mvt import MVTEEG
from eeg_mvt_dataset import EEGDataset

# Ignore RuntimeWarning and FutureWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Enable CUDA
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

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

# Create necessary directories
for dir_path in ['images_mvt_kfold', 'models', 'logs', 'models/checkpoints']:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Create log file with timestamp
log_filename = f'logs/training_log_{time.strftime("%Y%m%d_%H%M%S")}.txt'

def log_message(message, filename=log_filename):
    print(message)
    with open(filename, 'a') as f:
        f.write(message + '\n')

# Model parameters
num_chans = 19
num_classes = 3
dim = 512
dropout_rate = 0.1

def init_weights(m):
    """Fixed initialization function using relu instead of gelu"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Initialize model
mvt_model = MVTEEG(
    num_channels=num_chans,
    num_classes=num_classes,
    dim=dim
)
mvt_model.apply(init_weights)

# Log model architecture
log_message(str(mvt_model))
log_message(f"Model parameters: num_channels={num_chans}, num_classes={num_classes}, dim={dim}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mvt_model = mvt_model.to(device)

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

# Create dataset
train_dataset = EEGDataset(data_dir, balanced_train_data)

# Training parameters
epochs = 150
batch_size = 16
learning_rate = 0.0003  # Increased from 0.0001
accumulation_steps = 4
max_grad_norm = 1.0

# Resume training setup
checkpoint_dir = 'models/checkpoints'
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

# Training loop
for fold, (train_index, valid_index) in enumerate(skf.split(train_dataset.data, train_dataset.labels)):
    if fold < start_fold:
        continue
        
    log_message(f'\nFold {fold + 1}/5')
    
    # Reset model for each fold
    mvt_model.apply(init_weights)
    
    # Prepare data loaders
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
    
    log_message(f'Train dataloader: {len(train_dataloader)} batches')
    log_message(f'Valid dataloader: {len(valid_dataloader)} batches')
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        mvt_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,  # Reduced from 0.1
        betas=(0.9, 0.95)
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.2,  # Increased warm-up period
        div_factor=10,  # Reduced from 25
        final_div_factor=1e3,
    )
    
    # Load checkpoint if resuming
    if fold == start_fold and start_epoch > 0:
        mvt_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Initialize gradient scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Training epochs
    fold_losses = []
    fold_valid_losses = []
    
    for epoch in range(epochs):
        if fold == start_fold and epoch < start_epoch:
            continue
            
        start_time = time.time()
        mvt_model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        # Training loop
        for batch_idx, (inputs_dict, labels) in enumerate(train_dataloader):
            # Move data to device
            inputs = {k: v.to(device) for k, v in inputs_dict.items()}
            labels = labels.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = mvt_model(inputs['time'])
                loss = criterion(outputs, labels) / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(mvt_model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                if batch_idx % (accumulation_steps * 10) == 0:
                    log_message(f'Batch {batch_idx}/{len(train_dataloader)}, Gradient norm: {grad_norm:.4f}')
            
            train_loss += loss.item() * accumulation_steps
        
        # Validation
        mvt_model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs_dict, labels in valid_dataloader:
                inputs = {k: v.to(device) for k, v in inputs_dict.items()}
                labels = labels.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = mvt_model(inputs['time'])
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
        
        # Store losses
        fold_losses.append(epoch_train_loss)
        fold_valid_losses.append(epoch_valid_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Log metrics
        metrics_message = (
            f'Fold {fold + 1}/5, Epoch {epoch + 1}/{epochs}\n'
            f'Time: {epoch_time:.2f}s\n'
            f'Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}\n'
            f'Accuracy: {accuracy:.2f}%, F1: {f1:.4f}\n'
            f'Precision: {precision:.4f}, Recall: {recall:.4f}\n'
            f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}'
        )
        log_message(metrics_message)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'fold': fold,
            'model_state_dict': mvt_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': epoch_train_loss,
            'valid_loss': epoch_valid_loss,
            'accuracy': accuracy,
            'best_accuracy': best_accuracy,
            'train_losses': train_losses + fold_losses,
            'valid_losses': valid_losses + fold_valid_losses
        }
        
        torch.save(checkpoint, f'models/checkpoints/fold{fold+1}_epoch{epoch+1}.pt')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(mvt_model.state_dict(), f'models/best_mvt_model_fold{fold+1}.pth')
            log_message(f'New best model saved with accuracy: {accuracy:.2f}%')
    
    # Store fold results
    train_losses.extend(fold_losses)
    valid_losses.extend(fold_valid_losses)
    
    # Plot fold results
    plt.figure(figsize=(12, 6))
    plt.plot(fold_losses, label='Train Loss')
    plt.plot(fold_valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - Fold {fold+1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images_mvt_kfold/losses_fold{fold+1}.png')
    plt.close()
# Final results
log_message('\nTraining completed!')
log_message(f'Average Training Loss: {np.mean(train_losses):.4f}')
log_message(f'Average Validation Loss: {np.mean(valid_losses):.4f}')
log_message(f'Best Accuracy: {best_accuracy:.2f}%')

# Plot overall results
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Overall Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('images_mvt_kfold/losses_overall.png')
plt.close()

log_message('Training completed. Models and plots saved.')