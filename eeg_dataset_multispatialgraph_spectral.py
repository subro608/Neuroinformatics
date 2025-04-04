import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import mne
from scipy import signal
from scipy.spatial.distance import pdist, squareform
import warnings

# Suppress MNE and RuntimeWarnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class SpatialSpectralEEGDataset(Dataset):
    """
    Enhanced dataset class for loading EEG signals with improved spatial information 
    and spectral embeddings at multiple scales for the combined spatial-spectral model
    """
    def __init__(self, data_dir, data_info, scales=[3, 4, 5], time_length=8000, 
                augment=False, adjacency_type='distance'):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing the data files
            data_info: List of dictionaries with file and label info
            scales: List of scales for the spectral embeddings
            time_length: Maximum time length for EEG signals (will be padded/truncated)
            augment: Whether to apply data augmentation
            adjacency_type: Type of adjacency matrix ('distance', 'correlation', or 'combined')
        """
        self.data_dir = data_dir
        self.data_info = data_info
        self.scales = scales
        self.time_length = time_length
        self.augment = augment
        self.adjacency_type = adjacency_type
        
        # Extract file paths and labels
        self.data = [os.path.join(data_dir, info['file_name']) for info in data_info]
        
        # Convert label strings to integers
        label_map = {'A': 0, 'C': 1, 'F': 2}
        self.labels = [label_map[info['label']] for info in data_info]
        
        # Define spatial information - standard 10-20 electrode positions in 3D
        self.electrode_positions = {
            # Enhanced 3D positions (x, y, z coordinates)
            'Fp1': (-0.25, 0.9, 0.0), 'Fp2': (0.25, 0.9, 0.0),
            'F7': (-0.7, 0.65, 0.0), 'F3': (-0.4, 0.65, 0.0), 'Fz': (0.0, 0.65, 0.0), 
            'F4': (0.4, 0.65, 0.0), 'F8': (0.7, 0.65, 0.0),
            'T3': (-0.85, 0.0, 0.0), 'C3': (-0.4, 0.0, 0.1), 'Cz': (0.0, 0.0, 0.2), 
            'C4': (0.4, 0.0, 0.1), 'T4': (0.85, 0.0, 0.0),
            'T5': (-0.7, -0.65, 0.0), 'P3': (-0.4, -0.65, 0.0), 'Pz': (0.0, -0.65, 0.1), 
            'P4': (0.4, -0.65, 0.0), 'T6': (0.7, -0.65, 0.0),
            'O1': (-0.25, -0.9, 0.0), 'O2': (0.25, -0.9, 0.0)
        }
        
        # Create position matrix for easier computation
        self.position_matrix = np.array([pos for pos in self.electrode_positions.values()])
        
        # Generate the adjacency matrix
        self.adjacency = self._create_adjacency_matrix(self.adjacency_type)
        
        # Store channel names
        self.channel_names = list(self.electrode_positions.keys())
        
        # Store lobe groups for functional connectivity analysis
        self.lobe_groups = {
            'frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
            'central': ['C3', 'Cz', 'C4'],
            'temporal': ['T3', 'T4', 'T5', 'T6'],
            'parietal': ['P3', 'Pz', 'P4'],
            'occipital': ['O1', 'O2']
        }
        
        # Create lobe masks for faster access
        self.lobe_masks = self._create_lobe_masks()
        
        # Define frequency bands for spectral analysis
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        print(f"Enhanced dataset created with {len(self.data)} samples across scales {scales}")
        print(f"Using {adjacency_type} adjacency matrix")
    
    def _create_adjacency_matrix(self, adjacency_type):
        """
        Create adjacency matrix based on specified type
        
        Args:
            adjacency_type: Type of adjacency ('distance', 'correlation', or 'combined')
            
        Returns:
            np.ndarray: Adjacency matrix for EEG electrodes
        """
        num_channels = len(self.electrode_positions)
        
        if adjacency_type == 'distance':
            # Compute pairwise Euclidean distances
            distances = squareform(pdist(self.position_matrix))
            
            # Convert distances to adjacency (inverse relationship)
            adjacency = 1.0 / (1.0 + distances)
            np.fill_diagonal(adjacency, 1.0)  # Self-connection is 1
            
        elif adjacency_type == 'correlation':
            # Will be computed dynamically per sample
            adjacency = np.eye(num_channels)
            
        elif adjacency_type == 'combined':
            # Distance-based component
            distances = squareform(pdist(self.position_matrix))
            distance_adj = 1.0 / (1.0 + distances)
            np.fill_diagonal(distance_adj, 1.0)
            
            # Initialize correlation component (will be added dynamically per sample)
            adjacency = distance_adj
            
        else:
            # Default to distance-based
            distances = squareform(pdist(self.position_matrix))
            adjacency = 1.0 / (1.0 + distances)
            np.fill_diagonal(adjacency, 1.0)
        
        # Normalize adjacency matrix
        return adjacency / adjacency.max()
    
    def _create_lobe_masks(self):
        """
        Create binary masks for each lobe for faster access
        
        Returns:
            dict: Dictionary of binary masks for each lobe
        """
        masks = {}
        for lobe, channels in self.lobe_groups.items():
            mask = np.zeros(len(self.channel_names), dtype=bool)
            for ch in channels:
                if ch in self.channel_names:
                    idx = self.channel_names.index(ch)
                    mask[idx] = True
            masks[lobe] = mask
        return masks
    
    def _compute_dynamic_adjacency(self, eeg_data):
        """
        Compute dynamic adjacency matrix based on EEG data
        
        Args:
            eeg_data: EEG data of shape (channels, time)
            
        Returns:
            np.ndarray: Dynamic adjacency matrix
        """
        num_channels = eeg_data.shape[0]
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(eeg_data)
        
        # Handle NaN values that might occur
        correlation_matrix = np.nan_to_num(correlation_matrix)
        
        # Ensure positive values in adjacency (take absolute values)
        correlation_matrix = np.abs(correlation_matrix)
        
        if self.adjacency_type == 'correlation':
            # Use only correlation
            adjacency = correlation_matrix
            
        elif self.adjacency_type == 'combined':
            # Combine with distance-based adjacency
            distance_adj = self.adjacency.copy()
            
            # Weighted combination (0.5 * distance + 0.5 * correlation)
            adjacency = 0.5 * distance_adj + 0.5 * correlation_matrix
            
        else:
            # Use pre-computed distance-based adjacency
            adjacency = self.adjacency.copy()
        
        # Normalize
        return adjacency / adjacency.max()
    
    def compute_spectral_embeddings(self, eeg_data, sfreq=250):
        """
        Compute enhanced spectral embeddings at different scales with spatial awareness
        
        Args:
            eeg_data: EEG data of shape (channels, time)
            sfreq: Sampling frequency
            
        Returns:
            dict: Dictionary with spectral embeddings at different scales
        """
        num_channels = eeg_data.shape[0]
        spectral_embeddings = {}
        
        # Define scale-specific frequency bands
        scale_freqs = {
            3: [(0.5, 8), (8, 13), (13, 30)],                            # Basic: low, mid, high
            4: [(0.5, 4), (4, 8), (8, 13), (13, 30)],                    # Standard: delta, theta, alpha, beta
            5: [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)],          # Full: delta, theta, alpha, beta, gamma
            6: [(0.5, 2), (2, 4), (4, 8), (8, 13), (13, 30), (30, 50)],  # Detailed: subdivided delta, theta, alpha, beta, gamma
        }
        
        # Compute embeddings for each scale
        for scale in self.scales:
            # Get appropriate frequency bands for this scale
            if scale in scale_freqs:
                freqs = scale_freqs[scale]
            else:
                # Default to using the first N frequency bands
                freqs = list(self.frequency_bands.values())[:scale]
            
            # Initialize embeddings
            embeddings = np.zeros((num_channels, len(freqs)))
            
            # Compute power for each channel and band
            for ch_idx in range(num_channels):
                ch_data = eeg_data[ch_idx]
                
                # Skip if channel is all zeros
                if np.all(ch_data == 0):
                    continue
                
                # Compute power spectral density using Welch's method with more segments for better frequency resolution
                f, psd = signal.welch(ch_data, fs=sfreq, nperseg=min(512, len(ch_data)//4), 
                                      noverlap=min(256, len(ch_data)//8))
                
                # Extract power in each frequency band
                for band_idx, (fmin, fmax) in enumerate(freqs):
                    idx = np.logical_and(f >= fmin, f <= fmax)
                    if np.any(idx):
                        embeddings[ch_idx, band_idx] = np.mean(psd[idx])
            
            # Log transform to normalize power values (avoid log(0))
            embeddings = np.log(embeddings + 1e-10)
            
            # Add spatial weighting - give slightly more weight to frontal channels for AD detection
            if 'A' in [self.data_info[i]['label'] for i in range(len(self.data_info))]:
                # Get indices of frontal channels
                frontal_mask = self.lobe_masks['frontal']
                # Apply small weight increase (10%)
                embeddings[frontal_mask] *= 1.1
            
            # Store embeddings
            spectral_embeddings[f'scale_{scale}'] = embeddings
        
        return spectral_embeddings
    
    def apply_augmentation(self, eeg_data):
        """
        Apply data augmentation to EEG data
        
        Args:
            eeg_data: EEG data of shape (channels, time)
            
        Returns:
            np.ndarray: Augmented EEG data
        """
        if not self.augment:
            return eeg_data
        
        # Make a copy to avoid modifying the original
        augmented = eeg_data.copy()
        
        # Random augmentation selection
        aug_type = np.random.choice(['noise', 'scaling', 'channel_dropout', 'none'], p=[0.3, 0.2, 0.1, 0.4])
        
        if aug_type == 'noise':
            # Add small Gaussian noise
            noise_level = np.random.uniform(0.01, 0.05)
            augmented += np.random.normal(0, noise_level, augmented.shape)
            
        elif aug_type == 'scaling':
            # Random scaling
            scale_factor = np.random.uniform(0.9, 1.1)
            augmented *= scale_factor
            
        elif aug_type == 'channel_dropout':
            # Randomly zero out 1-3 channels
            num_channels = np.random.randint(1, 4)
            channels_to_drop = np.random.choice(augmented.shape[0], num_channels, replace=False)
            augmented[channels_to_drop] = 0
        
        return augmented
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset with enhanced spatial features
        
        Returns:
            dict: Dictionary containing:
                - 'raw_eeg': Raw EEG signal
                - 'scale_X': Spectral embedding at scale X for each scale in self.scales
                - 'adjacency': Adjacency matrix for EEG electrodes
                - 'spatial_positions': 3D positions of electrodes
            label: Class label as integer
        """
        file_path = self.data[idx]
        label = self.labels[idx]
        
        # Initialize sample data
        raw_eeg = None
        spectral_embeddings = {}
        dynamic_adjacency = None
        
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
                    
                    # Compute dynamic adjacency matrix based on data
                    if self.adjacency_type in ['correlation', 'combined']:
                        dynamic_adjacency = self._compute_dynamic_adjacency(eeg_data)
                    
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
                            # Create default embeddings if no data available
                            spectral_embeddings[scale_key] = torch.zeros(19, scale)
            
            # Apply data augmentation if enabled
            if self.augment:
                raw_eeg = self.apply_augmentation(raw_eeg)
            
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
        
        # Create the sample dictionary with spatial information
        sample = {
            'raw_eeg': torch.FloatTensor(raw_eeg),
            'adjacency': torch.FloatTensor(dynamic_adjacency if dynamic_adjacency is not None else self.adjacency),
            'spatial_positions': torch.FloatTensor(self.position_matrix)
        }
        
        # Add spectral embeddings to the sample
        sample.update(spectral_embeddings)
        
        return sample, label
    
    def get_adjacency(self):
        """Return the adjacency matrix for visualization"""
        return self.adjacency
    
    def get_electrode_positions(self):
        """Return electrode positions for visualization"""
        return self.electrode_positions
    
    def get_lobe_masks(self):
        """Return lobe masks for regional analysis"""
        return self.lobe_masks


# Function to create the dataset
def create_dataset(data_dir, data_info, scales=[3, 4, 5], time_length=8000, 
                   augment=False, adjacency_type='combined'):
    """Helper function to create the dataset with proper parameters"""
    return SpatialSpectralEEGDataset(
        data_dir=data_dir,
        data_info=data_info,
        scales=scales,
        time_length=time_length,
        augment=augment,
        adjacency_type=adjacency_type
    )