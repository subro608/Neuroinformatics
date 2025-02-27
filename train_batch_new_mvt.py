import os
import sys
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
import torch.multiprocessing as mp

# Add EEGPT path
sys.path.append(os.path.join(os.getcwd(), 'EEGPT'))
sys.path.append(os.path.join(os.getcwd(), 'EEGPT/downstream/Modules/models'))

# Import EEGPT model
from EEGPT.downstream.Modules.models.EEGPT_mcae_finetune import EEGPTClassifier

# Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Initialize settings
    set_seed(42)
    if not os.path.exists('images'):
        os.makedirs('images')

    # Model parameters
    num_channels = 19
    num_classes = 3
    dim = 128

    # Define channel names
    use_channels_names = [      
        'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'O1', 'O2'
    ]

    # Initialize EEGPT
    eegpt_model = EEGPTClassifier(
        num_classes=3,
        in_channels=num_channels,
        use_channels_names=use_channels_names,
        img_size=[len(use_channels_names), 2000],
        use_chan_conv=True,
        use_predictor=True
    )

    # Load EEGPT checkpoint
    eegpt_path = os.path.join('EEGPT', 'pretrain', 'eegpt_mcae_58chs_4s_large4E.ckpt')
    checkpoint = torch.load(eegpt_path, weights_only=True)
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    eegpt_model.load_state_dict(state_dict, strict=False)

    # Create MVT model
    from multimodal_mvt_model import create_multimodal_mvt
    mvt_model = create_multimodal_mvt(
        num_classes=num_classes,
        dim=dim,
        eegpt_model=eegpt_model
    )

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mvt_model.to(device)
    eegpt_model.to(device)

    # Data loading
    data_dir = 'model_data'
    with open(os.path.join(data_dir, 'labels.json'), 'r') as file:
        data_info = json.load(file)

    # Balance dataset
    train_data = [d for d in data_info if d['type'] == 'train']
    classes = ['A', 'C', 'F']
    class_data = {c: [d for d in train_data if d['label'] == c] for c in classes}
    
    min_samples = min(len(data) for data in class_data.values())
    balanced_data = []
    for c in classes:
        balanced_data.extend(random.sample(class_data[c], min_samples))
        print(f'Class {c}: {min_samples}')
    
    print(f'Total balanced dataset size: {len(balanced_data)}')

    # Create dataset and dataloader
    from multimodal_dataset import MultiModalDataset
    train_dataset = MultiModalDataset(data_dir, balanced_data)
    
    # Modified DataLoader with reduced number of workers and prefetch factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase gradually
        pin_memory=True
    )

    # Training parameters
    learning_rate = 0.001
    epochs = 100
    max_grad_norm = 1.0

    # Optimizer and scheduler
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

    criterion = nn.CrossEntropyLoss()

    # Training loop
    train_losses = []
    train_accuracies = []
    best_accuracy = 0.0

    print(f"Starting training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")

    try:
        for epoch in range(epochs):
            start_time = time.time()
            mvt_model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs_dict, labels) in enumerate(train_loader):
                # Move data to device
                inputs_dict = {
                    'eeg': {
                        'time': inputs_dict['eeg']['time'].to(device),
                        'freq': inputs_dict['eeg']['freq'].to(device),
                        'connectivity': inputs_dict['eeg']['connectivity'].to(device)
                    },
                    'mod1': inputs_dict['mod1'].to(device),
                    'mod2': inputs_dict['mod2'].to(device),
                    'mod3': inputs_dict['mod3'].to(device)
                }
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = mvt_model(inputs_dict)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(mvt_model.parameters(), max_grad_norm)
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if (batch_idx + 1) % 10 == 0:
                    print(f'Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

            scheduler.step()
            
            # Calculate epoch statistics
            epoch_loss = epoch_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                torch.save(mvt_model.state_dict(), 'best_multimodal_mvt.pth')

            print(f'\nEpoch {epoch + 1}/{epochs}')
            print(f'Train Loss: {epoch_loss:.4f}')
            print(f'Train Accuracy: {epoch_acc:.2f}%')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print(f'Time: {time.time() - start_time:.2f}s\n')

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

    # Save plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig('images/training_curves.png')
    plt.close()

    print(f'Best accuracy: {best_accuracy:.2f}%')
    print('Training complete! Model and plots saved.')

if __name__ == '__main__':
    mp.freeze_support()  # Add this line for Windows support
    main()