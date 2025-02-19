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
from eeg_mvt import MVTEEG
from eeg_mvt_dataset import EEGDataset

# Ignore RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Enable CUDA
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

# Set random seed
random.seed(42)

if not os.path.exists('images'):
    os.makedirs('images')

# Model params
num_chans = 19
num_classes = 3
dim = 512  # Transformer embedding dimension
dropout_rate = 0.5

# Model - modified to match MVTEEG class definition
mvt_model = MVTEEG(
    num_channels=num_chans,
    num_classes=num_classes,
    dim=dim
)

print(mvt_model)
print(f"Model params: num_channels={num_chans}, num_classes={num_classes}, dim={dim}")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mvt_model.to(device)

# Rest of your training script remains the same
data_dir = 'model-data'
data_file = 'labels.json'

with open(os.path.join(data_dir, data_file), 'r') as file:
    data_info = json.load(file)

train_data = [d for d in data_info if d['type'] == 'train']

# Separate training data by class
train_data_A = [d for d in train_data if d['label'] == 'A']
train_data_C = [d for d in train_data if d['label'] == 'C']
train_data_F = [d for d in train_data if d['label'] == 'F']

# Balance dataset
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

# Create dataset and dataloader
train_dataset = EEGDataset(data_dir, balanced_train_data)
indices = list(range(len(train_dataset)))
train_sampler = SubsetRandomSampler(indices)
train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)

print(f'Train dataset: {len(train_dataset)} samples')
print(f'Train dataloader: {len(train_dataloader)} batches')
print(f'Train dataloader batch size: {train_dataloader.batch_size}\n')

# Modified Training parameters
learning_rate = 0.007  # From 0.0007 (10x increase)
epochs = 100

warmup_steps = 1000

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
# Modified optimizer and scheduler
optimizer = optim.AdamW(
    mvt_model.parameters(),
    lr=learning_rate,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    epochs
)

# Add gradient clipping
max_grad_norm = 1.0

# Training loop with modifications
train_losses = []
best_accuracy = 0.0
for epoch in range(epochs):
    start_time = time.time()
    mvt_model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs_dict, labels) in enumerate(train_dataloader):
        # Move each tensor in the dictionary to device
        inputs = {k: v.to(device) for k, v in inputs_dict.items()}
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass using time-domain data
        outputs = mvt_model(inputs['time'])
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(mvt_model.parameters(), max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Scheduler step (now per batch)
        scheduler.step()

        # Calculate accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print batch progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_dataloader)}, '
                  f'Loss: {loss.item():.4f}')

    # Calculate epoch statistics
    epoch_loss = train_loss/len(train_dataloader)
    epoch_acc = 100.*correct/total
    train_losses.append(epoch_loss)

    # Save best model
    if epoch_acc > best_accuracy:
        best_accuracy = epoch_acc
        torch.save(mvt_model.state_dict(), 'best_mvt_model.pth')
    
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f'\nEpoch {epoch + 1}/{epochs}')
    print(f'Train Loss: {epoch_loss:.4f}')
    print(f'Train Accuracy: {epoch_acc:.2f}%')
    print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    print(f'Time: {epoch_time:.2f}s\n')

print('Training complete!')

# Plot losses with more detail
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.savefig('images/mvt_train_losses.png')
plt.close()

print(f'Best accuracy: {best_accuracy:.2f}%')
print('Final model and plots saved')
# Save model
model_file = 'mvt_eeg.pth'
torch.save(mvt_model.state_dict(), model_file)
print(f'Model saved to {model_file}')