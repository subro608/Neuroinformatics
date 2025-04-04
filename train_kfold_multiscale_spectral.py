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
import wandb  # Import wandb
import time
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# Import the updated model and dataset
from eeg_dataset_multiscale_spectral import MultiScaleSpectralEEGDataset
import eeg_multiscale_spectral  # Import the whole module

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

# Modified SimpleSpectralEEG class with corrected batch normalization
class FixedSimpleSpectralEEG(nn.Module):
    def __init__(self, num_channels=19, num_classes=3, dim=256, scale=5, num_layers=3):
        super().__init__()
        self.scale = scale
        
        # Project spectral embeddings to model dimension
        self.projection = nn.Linear(scale, dim)
        
        # Use correct dimension for batch norm (dim not num_channels)
        self.bn_projection = nn.BatchNorm1d(dim)
        
        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(num_channels, dim) * 0.02)
        
        # Transformer encoders
        self.encoder_layers = nn.ModuleList([
            eeg_multiscale_spectral.MVTEncoder(dim, num_heads=8) for _ in range(num_layers)
        ])
        
        # Classification head with batch normalization
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, num_classes)
        )
        
    def init_weights(self):
        """Initialize with Xavier uniform for better gradient flow"""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.apply(_init_weights)
    
    def forward(self, x=None, x_dict=None):
        """
        Forward pass with spectral embeddings
        
        Args:
            x: Direct spectral embedding input (batch, channels, scale)
            x_dict: Dictionary of spectral embeddings at different scales
        """
        # Get the right input
        if x_dict is not None and f'scale_{self.scale}' in x_dict:
            x = x_dict[f'scale_{self.scale}']
        elif x is None:
            raise ValueError("Either x_dict or x must be provided")
        
        batch_size, num_channels, _ = x.shape
        
        # Project to model dimension
        h = self.projection(x)  # (batch, channels, dim)
        
        # Apply batch normalization - this is the key change
        # Transpose, apply BN, transpose back
        h = h.transpose(1, 2)  # (batch, dim, channels)
        h = self.bn_projection(h)
        h = h.transpose(1, 2)  # (batch, channels, dim)
        
        # Add positional encoding
        h = h + self.pos_encoding.unsqueeze(0)  # (batch, channels, dim)
        
        # Prepare for transformer: (channels, batch, dim)
        h = h.permute(1, 0, 2)
        
        # Apply transformer
        for encoder in self.encoder_layers:
            h = encoder(h)
        
        # Global average pooling over channels
        h = h.mean(dim=0)  # (batch, dim)
        
        # Classification
        output = self.classifier(h)
        
        return output

def create_fixed_model(num_channels=19, num_classes=3, dim=256, scale=5, num_layers=3):
    """Create a model with fixed batch normalization"""
    model = FixedSimpleSpectralEEG(num_channels, num_classes, dim, scale, num_layers)
    model.init_weights()
    return model
def main():
    # Model and training parameters
    num_channels = 19
    num_classes = 3
    dim = 256
    scales = [3, 4, 5]  # Multi-scale approach
    
    # For simple model, always use num_clusters=5 to match the model's expectations
    simple_model_scale = 5
    
    # Use the multiscale model - set this to True to use the enhanced multiscale model
    use_multiscale = True
    model_type = 'multiscale' if use_multiscale else 'graph'
    
    batch_size = 16
    epochs = 300
    learning_rate = 0.001
    weight_decay = 0.0005
    accumulation_steps = 4
    max_grad_norm = 0.5
    
    # Initialize wandb
    wandb.init(
        # Track hyperparameters and run metadata
        config={
            "model_type": model_type,
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
            "multiscale": use_multiscale
        }
    )
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'spectral_{model_type}_{timestamp}'
    
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
    if use_multiscale:
        # For multiscale model, pass the list of scales instead of single scale
        model = eeg_multiscale_spectral.create_model(
            model_type=model_type,
            num_channels=num_channels, 
            num_classes=num_classes, 
            dim=dim,
            num_clusters=scales  # Pass all scales for multiscale model
        )
        log_message(f"Using MultiScale model with scales {scales}")
    else:
        # For single scale model, use the simple_model_scale
        model = eeg_multiscale_spectral.create_model(
            model_type=model_type,
            num_channels=num_channels, 
            num_classes=num_classes, 
            dim=dim,
            num_clusters=simple_model_scale
        )
        log_message(f"Using {model_type} model with scale {simple_model_scale}")
    
    # Log model architecture
    log_message(str(model))
    log_message(f"Model parameters: num_channels={num_channels}, num_classes={num_classes}, "
                f"dim={dim}, model_type={model_type}")
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
    log_message("Creating dataset with spectral features...")
    train_dataset = MultiScaleSpectralEEGDataset(data_dir, balanced_train_data, scales=scales)
    
    # Debug: print stats about the first batch to check data quality
    sample_batch, sample_labels = next(iter(DataLoader(train_dataset, batch_size=4)))
    
    log_message("\nData Statistics:")
    data_stats = {}
    for scale in scales:
        features = sample_batch[f'scale_{scale}']
        log_message(f"Scale {scale} - Shape: {features.shape}, "
                  f"Min: {features.min().item():.4f}, Max: {features.max().item():.4f}, "
                  f"Mean: {features.mean().item():.4f}, Std: {features.std().item():.4f}")
        data_stats[f"scale_{scale}_min"] = features.min().item()
        data_stats[f"scale_{scale}_max"] = features.max().item()
        data_stats[f"scale_{scale}_mean"] = features.mean().item()
        data_stats[f"scale_{scale}_std"] = features.std().item()
    
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
        if use_multiscale:
            # For multiscale model, pass the list of scales
            model = eeg_multiscale_spectral.create_model(
                model_type=model_type,
                num_channels=num_channels, 
                num_classes=num_classes, 
                dim=dim,
                num_clusters=scales  # Pass all scales for multiscale model
            )
        else:
            # For single scale model, use the simple_model_scale
            model = eeg_multiscale_spectral.create_model(
                model_type=model_type,
                num_channels=num_channels, 
                num_classes=num_classes, 
                dim=dim,
                num_clusters=simple_model_scale
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
            lr=learning_rate * 0.1,  # Reduce learning rate for AdamW
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
        
        # Initialize gradient scaler
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
                # For multiscale model, we'll use all scales
                if use_multiscale:
                    # Process all scales
                    scale_inputs = {k: v.to(device) for k, v in inputs_dict.items() if k.startswith('scale_')}
                    adjacency = inputs_dict['adjacency'].to(device)
                    labels = labels.to(device)
                    
                    # Forward pass with mixed precision
                    with torch.amp.autocast(device_type=device.type):
                        # Use embeddings_dict for multiscale model
                        outputs = model(embeddings_dict=scale_inputs, adjacency=adjacency)
                        loss = criterion(outputs, labels) / accumulation_steps
                else:
                    # For single scale model, use only the specified scale
                    single_scale_data = inputs_dict[f'scale_{simple_model_scale}'].to(device)
                    labels = labels.to(device)
                    
                    # Forward pass with mixed precision
                    with torch.amp.autocast(device_type=device.type):
                        # Pass the tensor data directly to the model
                        outputs = model(x=single_scale_data)
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
                    
                    # For multiscale model, we'll use all scales
                    if use_multiscale:
                        # Process all scales
                        scale_inputs = {k: v.to(device) for k, v in inputs_dict.items() if k.startswith('scale_')}
                        adjacency = inputs_dict['adjacency'].to(device)
                        labels = labels.to(device)
                        
                        with torch.amp.autocast(device_type=device.type):
                            outputs = model(embeddings_dict=scale_inputs, adjacency=adjacency)
                            loss = criterion(outputs, labels)
                    else:
                        # For single scale model, use only the specified scale
                        single_scale_data = inputs_dict[f'scale_{simple_model_scale}'].to(device)
                        labels = labels.to(device)
                        
                        with torch.amp.autocast(device_type=device.type):
                            outputs = model(x=single_scale_data)
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
            if accuracy > best_fold_accuracy:
                best_fold_accuracy = accuracy
                torch.save(model.state_dict(), f'{base_dir}/models/best_model_fold{fold+1}.pth')
                log_message(f'* New best model saved for fold {fold+1} with accuracy: {accuracy:.2f}% *')
                
                # Save the adjacency matrix for brain connectivity visualization
                if 'adjacency' in inputs_dict:
                    # Visualize brain connectivity
                    plt.figure(figsize=(10, 8))
                    plt.imshow(inputs_dict['adjacency'][0].cpu().numpy(), cmap='viridis')
                    plt.colorbar()
                    plt.title(f'EEG Channel Adjacency Matrix - Fold {fold+1}')
                    plt.savefig(f'{base_dir}/connectivity/adjacency_fold{fold+1}.png')
                    plt.close()
                    
                    # Log brain connectivity to wandb
                    adjacency_fig = plt.figure(figsize=(10, 8))
                    plt.imshow(inputs_dict['adjacency'][0].cpu().numpy(), cmap='viridis')
                    plt.colorbar()
                    plt.title(f'EEG Channel Adjacency Matrix - Fold {fold+1}')
                    wandb.log({f"connectivity_fold_{fold+1}": wandb.Image(adjacency_fig)})
                    plt.close()
                
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
                    
                    # Log confusion matrix to wandb
                    cm_fig = plt.figure(figsize=(10, 8))
                    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    plt.title(f'Confusion Matrix - Fold {fold+1} (Acc: {accuracy:.2f}%)')
                    plt.colorbar()
                    plt.xticks(tick_marks, class_names)
                    plt.yticks(tick_marks, class_names)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, format(cm[i, j], 'd'),
                                    horizontalalignment="center",
                                    color="white" if cm[i, j] > thresh else "black")
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    wandb.log({"confusion_matrix": wandb.Image(cm_fig)})
                    plt.close()
            
            # Log model performance over time
            wandb.log({
                "best_fold_accuracy": best_fold_accuracy,
                "best_overall_accuracy": best_accuracy
            })



if __name__ == "__main__":
    main()