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

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

from eeg_mvt import MVTEEG
from eeg_mvt_dataset import EEGDataset

# Ignore RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Enable CUDA
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists('images_mvt_kfold'):
    os.makedirs('images_mvt_kfold')
if not os.path.exists('models'):
    os.makedirs('models')

# Model params
num_chans = 19
num_classes = 3
dim = 256  # Reduced from 512
num_heads = 8  # Reduced number of attention heads
dropout_rate = 0.1

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# Model initialization
mvt_model = MVTEEG(
    num_channels=num_chans,
    num_classes=num_classes,
    dim=dim
)
mvt_model.apply(init_weights)

print(mvt_model)
print(f"Model params: num_channels={num_chans}, num_classes={num_classes}, dim={dim}")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mvt_model = mvt_model.to(device)

# Data loading
data_dir = 'model_data'
data_file = 'labels.json'

with open(os.path.join(data_dir, data_file), 'r') as file:
    data_info = json.load(file)

train_data = [d for d in data_info if d['type'] == 'train']

# Balance dataset
train_data_A = [d for d in train_data if d['label'] == 'A']
train_data_C = [d for d in train_data if d['label'] == 'C']
train_data_F = [d for d in train_data if d['label'] == 'F']

min_samples = min((len(train_data_A)+len(train_data_C))/2, 
                 (len(train_data_A)+len(train_data_F))/2,
                 (len(train_data_C)+len(train_data_F))/2)

a_index = int(min(min_samples, len(train_data_A)))
c_index = int(min(min_samples, len(train_data_C)))
f_index = int(min(min_samples, len(train_data_F)))

balanced_train_data = (random.sample(train_data_A, a_index) +
                      random.sample(train_data_C, c_index) +
                      random.sample(train_data_F, f_index))

print(f'Before Balancing\nA: {len(train_data_A)}, C: {len(train_data_C)}, F: {len(train_data_F)}')
print(f'After Balancing\nA: {a_index}, C: {c_index}, F: {f_index}')
print(f'Total: {len(balanced_train_data)}')

# Create dataset
train_dataset = EEGDataset(data_dir, balanced_train_data)

# Training parameters
epochs = 100
batch_size = 32
learning_rate = 0.0001
accumulation_steps = 4
max_grad_norm = 1.0

# Initialize k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Training components
criterion = nn.CrossEntropyLoss()
train_losses = []
best_accuracy = 0.0

for fold, (train_index, valid_index) in enumerate(skf.split(train_dataset.data, train_dataset.labels)):
    print(f'\nFold {fold + 1}/5')
    
    # Reset model for each fold
    mvt_model.apply(init_weights)
    
    # Prepare data loaders
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
    
    print(f'Train dataloader: {len(train_dataloader)} batches')
    print(f'Valid dataloader: {len(valid_dataloader)} batches\n')
    
    # Initialize optimizer and schedulers
    optimizer = optim.AdamW(
        mvt_model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1e4,
    )
    
    # Initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    
    fold_losses = []
    
    for epoch in range(epochs):
        start_time = time.time()
        mvt_model.train()
        train_loss = 0.0
        
        # Training loop
        for batch_idx, (inputs_dict, labels) in enumerate(train_dataloader):
            inputs = {k: v.to(device) for k, v in inputs_dict.items()}
            labels = labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = mvt_model(inputs['time'])
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(mvt_model.parameters(), max_grad_norm)
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
        
        with torch.no_grad():
            for inputs_dict, labels in valid_dataloader:
                inputs = {k: v.to(device) for k, v in inputs_dict.items()}
                labels = labels.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = mvt_model(inputs['time'])
                    loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate metrics
        epoch_train_loss = train_loss / len(train_dataloader)
        epoch_valid_loss = valid_loss / len(valid_dataloader)
        accuracy = 100. * correct / total
        fold_losses.append(epoch_train_loss)
        
        end_time = time.time()
        epoch_time = end_time - start_time

        print(f'Fold {fold + 1}/5, Epoch {epoch + 1}/{epochs}, '
              f'Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s, '
              f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(mvt_model.state_dict(), f'models/best_mvt_model_fold{fold+1}.pth')
    
    train_losses.extend(fold_losses)
    
    # Save fold model
    torch.save(mvt_model.state_dict(), f'models/mvt_model_fold{fold+1}.pth')
    
    # Plot fold losses
    plt.figure(figsize=(12, 6))
    plt.plot(fold_losses, label=f'Fold {fold+1} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - Fold {fold+1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images_mvt_kfold/train_losses_fold{fold+1}.png')
    plt.close()

# Calculate and print average loss
avg_train_loss = sum(train_losses) / len(train_losses)
print(f'\nTraining completed!')
print(f'Average Training Loss: {avg_train_loss:.4f}')
print(f'Best Accuracy: {best_accuracy:.2f}%')

# Plot overall training loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Overall Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('images_mvt_kfold/train_losses_overall.png')
plt.close()

print('Final models and plots saved.')