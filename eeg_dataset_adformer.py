import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import mne
import warnings
import random

# Suppress MNE and RuntimeWarnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class VanillaADformerEEGDataset(Dataset):
    """
    Dataset class for loading EEG signals specifically for vanilla ADformer architecture
    without additional adjacency matrices or spectral features.
    """
    def __init__(self, data_dir, data_info, time_length=2000, 
                 patch_lengths=[200, 400, 800], up_dimensions=[10, 15, 20],
                 augmentations=['none', 'jitter', 'scale'], normalize=True,
                 is_training=True):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing the data files
            data_info: List of dictionaries with file and label info
            time_length: Maximum time length for EEG signals (will be padded/truncated)
            patch_lengths: List of patch lengths for temporal dimension in ADformer
            up_dimensions: List of up-dimensions for channel dimension in ADformer
            augmentations: List of augmentation techniques to apply
            normalize: Whether to normalize the data
            is_training: Whether this is a training dataset (affects augmentation)
        """
        self.data_dir = data_dir
        self.data_info = data_info
        self.time_length = time_length
        self.patch_lengths = patch_lengths
        self.up_dimensions = up_dimensions
        self.augmentations = augmentations
        self.normalize = normalize
        self.is_training = is_training
        
        # Extract file paths and labels
        self.data = [os.path.join(data_dir, info['file_name']) for info in data_info]
        
        # Convert label strings to integers
        label_map = {'A': 0, 'C': 1, 'F': 2}
        self.labels = [label_map[info['label']] for info in data_info]
        
        # Initialize normalization parameters
        self.mean = None
        self.std = None
        
        print(f"Vanilla ADformer EEG dataset created with {len(self.data)} samples")
        print(f"Temporal patches: {patch_lengths}")
        print(f"Channel dimensions: {up_dimensions}")
        print(f"Augmentation methods: {augmentations}")
        print(f"Is training dataset: {is_training}")
    
    def set_normalization_stats(self, mean, std):
        """Set normalization statistics externally"""
        self.mean = mean
        self.std = std
        print(f"Normalization statistics set externally: mean shape {mean.shape}, std shape {std.shape}")
    
    def apply_augmentation(self, data, aug_type):
        """Apply augmentation to the data"""
        # Create a copy to avoid modifying the original
        aug_data = data.copy()
        
        if aug_type == 'none' or np.all(data == 0):
            return aug_data
        
        if aug_type == 'jitter':
            # Add random noise with smaller magnitude
            noise = np.random.normal(0, 0.005, data.shape)
            aug_data = aug_data + noise
        
        elif aug_type == 'scale':
            # Scale by a random factor - more conservative range
            scaling_factor = np.random.uniform(0.9, 1.1)
            aug_data = aug_data * scaling_factor
        
        elif aug_type == 'time_shift':
            # Shift time series slightly
            shift = random.randint(-100, 100)
            if shift > 0:
                aug_data = np.concatenate([np.zeros_like(aug_data[:shift]), aug_data[:-shift]], axis=0)
            elif shift < 0:
                aug_data = np.concatenate([aug_data[-shift:], np.zeros_like(aug_data[:shift])], axis=0)
        
        elif aug_type == 'permutation':
            # Permute small segments
            seg_len = min(200, data.shape[0] // 10)
            if seg_len > 0:
                for ch_idx in range(data.shape[1]):
                    segments = data.shape[0] // seg_len
                    for i in range(segments):
                        if random.random() < 0.1:  # 10% chance to permute (reduced from 20%)
                            j = random.randint(0, segments-1)
                            start_i, end_i = i*seg_len, (i+1)*seg_len
                            start_j, end_j = j*seg_len, (j+1)*seg_len
                            temp = aug_data[start_i:end_i, ch_idx].copy()
                            aug_data[start_i:end_i, ch_idx] = aug_data[start_j:end_j, ch_idx]
                            aug_data[start_j:end_j, ch_idx] = temp
        
        elif aug_type == 'masking':
            # Randomly mask segments - reduced masking
            mask_len = min(50, data.shape[0] // 40)
            if mask_len > 0:
                for ch_idx in range(data.shape[1]):
                    if random.random() < 0.1:  # 10% chance to apply masking (reduced from 30%)
                        start = random.randint(0, data.shape[0] - mask_len)
                        aug_data[start:start+mask_len, ch_idx] = 0
        
        return aug_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset with features formatted for ADformer"""
        file_path = self.data[idx]
        label = self.labels[idx]
        
        try:
            # Load EEG data
            raw_eeg = None
            
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if 'eeg' in data:
                    raw_eeg = data['eeg']
            
            elif file_path.endswith('.set'):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
                eeg_data = raw.get_data()
                
                # Use first 19 channels or pad if fewer
                if len(eeg_data) >= 19:
                    raw_eeg = eeg_data[:19]
                else:
                    padding = np.zeros((19 - len(eeg_data), eeg_data.shape[1]))
                    raw_eeg = np.vstack([eeg_data, padding])
            
            # Handle missing data or fallback
            if raw_eeg is None:
                raw_eeg_path = file_path.replace(os.path.splitext(file_path)[1], '_raw.npy')
                if os.path.exists(raw_eeg_path):
                    raw_eeg = np.load(raw_eeg_path)
                else:
                    raw_eeg = np.zeros((19, self.time_length))
            
            # Process raw EEG - resize to desired length
            if raw_eeg.shape[1] > self.time_length:
                raw_eeg = raw_eeg[:, :self.time_length]
            elif raw_eeg.shape[1] < self.time_length:
                padding = np.zeros((raw_eeg.shape[0], self.time_length - raw_eeg.shape[1]))
                raw_eeg = np.concatenate([raw_eeg, padding], axis=1)
            
            # Apply augmentation with reduced probability during training
            if self.is_training and len(self.augmentations) > 0 and random.random() < 0.3:  # Reduced from 0.5
                # Randomly select an augmentation
                aug_type = random.choice(self.augmentations)
                # Apply augmentation to transposed data (raw_eeg is [channels, time])
                augmented = self.apply_augmentation(raw_eeg.T, aug_type)
                # Return to original shape [channels, time]
                raw_eeg = augmented.T
            
            # Normalize data if requested and stats are available
            if self.normalize and not np.all(raw_eeg == 0):
                if self.mean is not None and self.std is not None:
                    # Use pre-computed stats for consistent normalization
                    raw_eeg = (raw_eeg - self.mean[:, np.newaxis]) / self.std[:, np.newaxis]
                else:
                    # If no global stats, normalize per channel
                    std = raw_eeg.std(axis=1, keepdims=True)
                    std[std < 1e-6] = 1.0
                    raw_eeg = (raw_eeg - raw_eeg.mean(axis=1, keepdims=True)) / std
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raw_eeg = np.zeros((19, self.time_length))
        
        # Format data for ADformer's expected input
        # ADformer expects: (seq_len, enc_in) -> will be batched to (batch_size, seq_len, enc_in)
        # Our data is: (channels, time) -> need to transpose to (time, channels)
        adformer_formatted_eeg = torch.FloatTensor(raw_eeg).transpose(0, 1)  # [seq_len, enc_in]
        
        # Create empty marker vector (for time features, optional in ADformer)
        mark_vector = torch.zeros(adformer_formatted_eeg.shape[0], 1)
        
        # Return the simplified sample - only what ADformer needs
        sample = {
            'eeg': adformer_formatted_eeg,
            'mark': mark_vector,
        }
        
        return sample, label
    
    def get_batch_for_adformer(self, batch_data):
        """
        Preprocess a batch of data for ADformer model - simplified for vanilla ADformer
        
        Args:
            batch_data: A batch of data from the dataloader
        
        Returns:
            Processed data ready for ADformer input
        """
        samples, labels = batch_data
        
        # Format according to ADformer input expectations
        batch_size = len(samples['eeg'])
        
        # Stack into batches
        # ADformer expects: x_enc [batch_size, seq_len, enc_in]
        x_enc = torch.stack([sample for sample in samples['eeg']])
        
        # ADformer expects: x_mark_enc [batch_size, seq_len, mark_size]
        x_mark_enc = torch.stack([sample for sample in samples['mark']])
        
        # ADformer provides dummy placeholders for decoder inputs in classification task
        x_dec = torch.zeros_like(x_enc)
        x_mark_dec = torch.zeros_like(x_mark_enc)
        
        # For the classification task, ADformer only uses x_enc and x_mark_enc
        # The decoder inputs are ignored but need to be provided
        adformer_inputs = {
            'x_enc': x_enc,
            'x_mark_enc': x_mark_enc,
            'x_dec': x_dec,
            'x_mark_dec': x_mark_dec,
            'labels': torch.tensor(labels)
        }
        
        return adformer_inputs, {}  # No additional data needed for vanilla ADformer


def create_vanilla_adformer_dataset(data_dir, data_info, time_length=2000, 
                                   patch_lengths=[200, 400, 800], 
                                   up_dimensions=[10, 15, 20], 
                                   augmentations=['none', 'jitter', 'scale'],
                                   normalize=True,
                                   is_training=True):
    """Helper function to create the vanilla ADformer dataset"""
    return VanillaADformerEEGDataset(
        data_dir=data_dir,
        data_info=data_info,
        time_length=time_length,
        patch_lengths=patch_lengths,
        up_dimensions=up_dimensions,
        augmentations=augmentations,
        normalize=normalize,
        is_training=is_training
    )


# Collate function - moved outside dataloader function to avoid pickling issues
def vanilla_adformer_collate_fn(batch):
    """Custom collate function simplified for vanilla ADformer"""
    samples = {
        'eeg': [item[0]['eeg'] for item in batch],
        'mark': [item[0]['mark'] for item in batch],
    }
    
    labels = [item[1] for item in batch]
    return samples, labels


def create_vanilla_adformer_dataloader(dataset, batch_size=32, shuffle=True, sampler=None, num_workers=0):
    """Create a dataloader for vanilla ADformer"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,  # Must be False when using a sampler
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=vanilla_adformer_collate_fn,
        pin_memory=torch.cuda.is_available()
    )