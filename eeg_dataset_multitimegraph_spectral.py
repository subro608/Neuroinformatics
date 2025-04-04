import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import mne
from scipy import signal
import warnings

# Suppress MNE and RuntimeWarnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class MultiScaleTimeSpectralEEGDataset(Dataset):
    """
    Dataset class for loading both raw EEG signals (time domain) and 
    spectral embeddings at multiple scales for the combined model
    """
    def __init__(self, data_dir, data_info, scales=[3, 4, 5], time_length=8000):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing the data files
            data_info: List of dictionaries with file and label info
            scales: List of scales for the spectral embeddings
            time_length: Maximum time length for raw EEG signals (will be padded/truncated)
        """
        self.data_dir = data_dir
        self.data_info = data_info
        self.scales = scales
        self.time_length = time_length
        
        # Extract file paths and labels
        self.data = [os.path.join(data_dir, info['file_name']) for info in data_info]
        
        # Convert label strings to integers
        label_map = {'A': 0, 'C': 1, 'F': 2}
        self.labels = [label_map[info['label']] for info in data_info]
        
        # Precomputed adjacency matrix for EEG electrode connectivity
        electrode_positions = {
            # Standard 10-20 electrode positions (x, y coordinates)
            'Fp1': (-0.25, 0.9), 'Fp2': (0.25, 0.9),
            'F7': (-0.7, 0.65), 'F3': (-0.4, 0.65), 'Fz': (0.0, 0.65), 'F4': (0.4, 0.65), 'F8': (0.7, 0.65),
            'T3': (-0.85, 0.0), 'C3': (-0.4, 0.0), 'Cz': (0.0, 0.0), 'C4': (0.4, 0.0), 'T4': (0.85, 0.0),
            'T5': (-0.7, -0.65), 'P3': (-0.4, -0.65), 'Pz': (0.0, -0.65), 'P4': (0.4, -0.65), 'T6': (0.7, -0.65),
            'O1': (-0.25, -0.9), 'O2': (0.25, -0.9)
        }
        
        # Compute distances between electrodes
        positions = list(electrode_positions.values())
        positions = np.array(positions)
        
        # Compute pairwise Euclidean distances
        num_channels = len(positions)
        distances = np.zeros((num_channels, num_channels))
        for i in range(num_channels):
            for j in range(num_channels):
                distances[i, j] = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
        
        # Convert distances to adjacency (inverse relationship)
        adjacency = 1.0 / (1.0 + distances)
        np.fill_diagonal(adjacency, 1.0)  # Self-connection is 1
        
        # Normalize adjacency matrix
        self.adjacency = adjacency / adjacency.max()
        
        # Set standard EEG channel names
        self.channel_names = list(electrode_positions.keys())
        
        print(f"Dataset created with {len(self.data)} samples across scales {scales}")
    
    def compute_spectral_embeddings(self, eeg_data, sfreq=250):
        """
        Compute spectral embeddings at different scales
        
        Args:
            eeg_data: EEG data of shape (channels, time)
            sfreq: Sampling frequency
            
        Returns:
            dict: Dictionary with spectral embeddings at different scales
        """
        num_channels = eeg_data.shape[0]
        spectral_embeddings = {}
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Compute power spectral density
        for scale in self.scales:
            if scale == 3:
                # Basic 3-scale: low, mid, high frequency power
                freqs = [(0.5, 8), (8, 13), (13, 30)]
            elif scale == 4:
                # 4-scale: delta, theta, alpha, beta
                freqs = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
            elif scale == 5:
                # 5-scale: delta, theta, alpha, beta, gamma
                freqs = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
            else:
                # Default: use standard frequency bands
                freqs = list(bands.values())[:scale]
            
            # Compute power in each frequency band for each channel
            embeddings = np.zeros((num_channels, len(freqs)))
            
            for ch_idx in range(num_channels):
                ch_data = eeg_data[ch_idx]
                
                # Compute power spectral density using Welch's method
                f, psd = signal.welch(ch_data, fs=sfreq, nperseg=min(256, len(ch_data)))
                
                # Extract power in each frequency band
                for band_idx, (fmin, fmax) in enumerate(freqs):
                    idx = np.logical_and(f >= fmin, f <= fmax)
                    if np.any(idx):
                        embeddings[ch_idx, band_idx] = np.mean(psd[idx])
            
            # Log transform to normalize power values
            embeddings = np.log(embeddings + 1e-10)
            
            # Store embeddings
            spectral_embeddings[f'scale_{scale}'] = embeddings
        
        return spectral_embeddings
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            dict: Dictionary containing:
                - 'raw_eeg': Raw EEG signal (time domain)
                - 'scale_X': Spectral embedding at scale X for each scale in self.scales
                - 'adjacency': Adjacency matrix for EEG electrodes
            label: Class label as integer
        """
        file_path = self.data[idx]
        label = self.labels[idx]
        
        # Initialize sample data
        raw_eeg = None
        spectral_embeddings = {}
        
        try:
            # Check file type and load accordingly
            if file_path.endswith('.pkl'):
                # Load from pickle file
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract raw EEG and spectral embeddings
                if 'eeg' in data:
                    raw_eeg = data['eeg']
                
                # Get pre-computed spectral embeddings
                for scale in self.scales:
                    scale_key = f'scale_{scale}'
                    if scale_key in data:
                        spectral_embeddings[scale_key] = torch.FloatTensor(data[scale_key])
            
            elif file_path.endswith('.set'):
                # Load EEGLAB file using MNE
                try:
                    # Use MNE to load EEGLAB file with minimal verbosity
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
                    
                    # Extract channel data
                    eeg_data = raw.get_data()
                    
                    # Select standard channels if available, otherwise use all channels
                    if len(eeg_data) >= 19:
                        # If we have more channels than needed, try to select standard ones
                        if set(self.channel_names).issubset(set(raw.ch_names)):
                            # Get indices of standard channels
                            ch_indices = [raw.ch_names.index(ch) for ch in self.channel_names 
                                         if ch in raw.ch_names]
                            eeg_data = eeg_data[ch_indices]
                        else:
                            # Just take the first 19 channels
                            eeg_data = eeg_data[:19]
                    
                    # Ensure we have 19 channels (pad if needed)
                    if len(eeg_data) < 19:
                        padding = np.zeros((19 - len(eeg_data), eeg_data.shape[1]))
                        eeg_data = np.vstack([eeg_data, padding])
                    
                    # Set raw EEG data
                    raw_eeg = eeg_data
                    
                    # Compute spectral embeddings on the fly
                    computed_embeddings = self.compute_spectral_embeddings(
                        eeg_data, sfreq=raw.info['sfreq']
                    )
                    
                    # Convert to torch tensors
                    for scale_key, embedding in computed_embeddings.items():
                        spectral_embeddings[scale_key] = torch.FloatTensor(embedding)
                    
                except Exception as e:
                    print(f"Error loading EEGLAB file {file_path}: {e}")
                    raw_eeg = np.zeros((19, self.time_length))
                    spectral_embeddings = {
                        f'scale_{scale}': torch.zeros(19, scale) for scale in self.scales
                    }
            
            # Fall back to .npy files if needed
            if raw_eeg is None:
                raw_eeg_path = file_path.replace(os.path.splitext(file_path)[1], '_raw.npy')
                if os.path.exists(raw_eeg_path):
                    raw_eeg = np.load(raw_eeg_path)
                else:
                    raw_eeg = np.zeros((19, self.time_length))
            
            # Check for each scale embedding
            for scale in self.scales:
                scale_key = f'scale_{scale}'
                if scale_key not in spectral_embeddings:
                    # Try to load from separate file
                    scale_path = file_path.replace(os.path.splitext(file_path)[1], f'_{scale}.npy')
                    if os.path.exists(scale_path):
                        spectral_data = np.load(scale_path)
                        spectral_embeddings[scale_key] = torch.FloatTensor(spectral_data)
                    else:
                        # If we have raw EEG but no embeddings, compute them
                        if raw_eeg is not None and not np.all(raw_eeg == 0):
                            if scale_key not in spectral_embeddings:
                                # Compute embeddings if we haven't already
                                if not spectral_embeddings:
                                    computed_embeddings = self.compute_spectral_embeddings(raw_eeg)
                                    for k, v in computed_embeddings.items():
                                        spectral_embeddings[k] = torch.FloatTensor(v)
                        else:
                            # Create random embeddings if no data available
                            spectral_embeddings[scale_key] = torch.zeros(19, scale)
            
            # Process raw EEG signal
            # Pad or truncate to the desired length
            if raw_eeg.shape[1] > self.time_length:
                raw_eeg = raw_eeg[:, :self.time_length]
            elif raw_eeg.shape[1] < self.time_length:
                padding = np.zeros((raw_eeg.shape[0], self.time_length - raw_eeg.shape[1]))
                raw_eeg = np.concatenate([raw_eeg, padding], axis=1)
            
            # Normalize raw EEG (if not all zeros)
            if not np.all(raw_eeg == 0):
                std = raw_eeg.std(axis=1, keepdims=True)
                # Avoid division by zero
                std[std < 1e-6] = 1.0
                raw_eeg = (raw_eeg - raw_eeg.mean(axis=1, keepdims=True)) / std
            
            # Normalize spectral embeddings (if not all zeros)
            for scale_key in spectral_embeddings:
                embed = spectral_embeddings[scale_key].numpy()
                if not np.all(embed == 0):
                    std = embed.std(axis=1, keepdims=True)
                    # Avoid division by zero
                    std[std < 1e-6] = 1.0
                    embed = (embed - embed.mean(axis=1, keepdims=True)) / std
                    spectral_embeddings[scale_key] = torch.FloatTensor(embed)
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            # Create dummy data in case of error
            raw_eeg = np.zeros((19, self.time_length))
            spectral_embeddings = {f'scale_{scale}': torch.zeros(19, scale) for scale in self.scales}
        
        # Create the sample dictionary
        sample = {
            'raw_eeg': torch.FloatTensor(raw_eeg),
            'adjacency': torch.FloatTensor(self.adjacency)
        }
        
        # Add spectral embeddings to the sample
        sample.update(spectral_embeddings)
        
        return sample, label
    
    def get_adjacency(self):
        """Return the adjacency matrix for visualization"""
        return self.adjacency