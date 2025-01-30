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

# Set random seed
random.seed(42)

if not os.path.exists('images_mvt_kfold'):
    os.makedirs('images_mvt_kfold')

# Model params
num_chans = 19
num_classes = 3
dim = 512  # Transformer embedding dimension
dropout_rate = 0.5

# Model
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

# Data
data_dir = 'model_data'
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

# Create dataset
train_dataset = EEGDataset(data_dir, balanced_train_data)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Training parameters
epochs = 810
learning_rate = 0.007

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    mvt_model.parameters(),
    lr=learning_rate,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

# Add gradient clipping
max_grad_norm = 1.0

train_losses = []
best_accuracy = 0.0

for fold, (train_index, valid_index) in enumerate(skf.split(train_dataset.data, train_dataset.labels)):
    print(f'\nFold {fold + 1}/5')

    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    valid_dataloader = DataLoader(train_dataset, batch_size=16, sampler=valid_sampler)

    print(f'Train dataloader: {len(train_dataloader)} batches')
    print(f'Valid dataloader: {len(valid_dataloader)} batches\n')

    for epoch in range(epochs):
        start_time = time.time()
        mvt_model.train()
        train_loss = 0.0
        
        # Training loop
        for batch_idx, (inputs_dict, labels) in enumerate(train_dataloader):
            # Move each tensor in the dictionary to device
            inputs = {k: v.to(device) for k, v in inputs_dict.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = mvt_model(inputs['time'])
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(mvt_model.parameters(), max_grad_norm)
            
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        mvt_model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs_dict, labels in valid_dataloader:
                inputs = {k: v.to(device) for k, v in inputs_dict.items()}
                labels = labels.to(device)
                outputs = mvt_model(inputs['time'])
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        accuracy = 100. * correct / total
        epoch_loss = train_loss / len(train_dataloader)
        valid_loss = valid_loss / len(valid_dataloader)
        train_losses.append(epoch_loss)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f'Fold {fold + 1}/5, Epoch {epoch + 1}/{epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s, '
              f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(mvt_model.state_dict(), 'best_mvt_model_kfold.pth')

# Calculate and print average loss
avg_train_loss = sum(train_losses) / len(train_losses)
print(f'Average Training Loss: {avg_train_loss:.4f}')
print(f'Best Accuracy: {best_accuracy:.2f}%')

# Save final model
model_file = 'models/mvt_5fold.pth'
torch.save(mvt_model.state_dict(), model_file)
print(f'Model saved to {model_file}')

# Plot overall training loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Overall Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('images_mvt_kfold/train_losses_5fold.png')
plt.close()

# Plot individual fold losses
for fold in range(5):
    plt.figure(figsize=(12, 6))
    fold_losses = train_losses[fold*epochs:(fold+1)*epochs]
    plt.plot(fold_losses, label=f'Fold {fold+1} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - Fold {fold+1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images_mvt_kfold/train_losses_fold{fold+1}.png')
    plt.close()

print('Training complete! Loss plots saved.')