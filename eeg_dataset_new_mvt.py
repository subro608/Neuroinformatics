import os
import mne
import warnings
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from scipy import signal
from sklearn.preprocessing import StandardScaler

# Ignore RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Enable CUDA for MNE
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

class MultiModalProcessor:
    """Handles preprocessing for different modalities"""
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.eeg_scaler = StandardScaler()
        self.other_scalers = {
            'mod1': StandardScaler(),
            'mod2': StandardScaler(),
            'mod3': StandardScaler()
        }

    def get_frequency_features(self, eeg_data):
        """Extract frequency domain features from EEG"""
        fft_data = np.abs(np.fft.fft(eeg_data, axis=1))
        
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        band_powers = np.zeros((eeg_data.shape[0], len(freq_bands)))
        freqs = np.fft.fftfreq(eeg_data.shape[1], 1/self.sampling_rate)
        
        for i, (band, (fmin, fmax)) in enumerate(freq_bands.items()):
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            band_powers[:, i] = np.mean(fft_data[:, band_mask], axis=1)
        
        return {
            'fft': fft_data,
            'band_powers': band_powers
        }

    def get_connectivity_features(self, eeg_data):
        """Extract connectivity features between channels"""
        n_channels = eeg_data.shape[0]
        connectivity = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    # Calculate PLV (Phase Locking Value)
                    analytic1 = signal.hilbert(eeg_data[i])
                    analytic2 = signal.hilbert(eeg_data[j])
                    phase_diff = np.angle(analytic1) - np.angle(analytic2)
                    connectivity[i,j] = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return connectivity

    def preprocess_eeg(self, eeg_data, normalize=True):
        """Preprocess EEG data"""
        if normalize:
            eeg_data = self.eeg_scaler.fit_transform(eeg_data.T).T
        
        freq_features = self.get_frequency_features(eeg_data)
        conn_features = self.get_connectivity_features(eeg_data)
        
        return {
            'time': eeg_data,
            'freq': freq_features,
            'connectivity': conn_features
        }

    def preprocess_modality(self, data, modality_name, normalize=True):
        """Preprocess other modality data"""
        if normalize:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            data = self.other_scalers[modality_name].fit_transform(data)
        return data

class MultiModalDataset(Dataset):
    def __init__(self, data_directory, dataset_info, processor=None):
        self.data_directory = data_directory
        self.dataset_info = dataset_info
        self.processor = processor if processor else MultiModalProcessor()
        self.labels = [d['label'] for d in dataset_info]

    def __len__(self):
        return len(self.dataset_info)

    def load_eeg_data(self, file_path):
        """Load EEG data from file"""
        # Assuming EEGLAB format
        raw = mne.io.read_raw_eeglab(file_path)
        return raw.get_data()

    def load_modality_data(self, file_info, modality_name):
        """Load data for a specific modality"""
        file_path = os.path.join(self.data_directory, 
                                file_info[f'{modality_name}_file'])
        # Add your specific loading logic here for each modality
        data = np.load(file_path)  # Example loading - modify as needed
        return data

    def __getitem__(self, idx):
        file_info = self.dataset_info[idx]
        
        # Load and preprocess EEG
        eeg_path = os.path.join(self.data_directory, file_info['eeg_file'])
        eeg_data = self.load_eeg_data(eeg_path)
        eeg_features = self.processor.preprocess_eeg(eeg_data)
        
        # Load and preprocess other modalities
        modalities = {}
        for mod_name in ['mod1', 'mod2', 'mod3']:
            mod_data = self.load_modality_data(file_info, mod_name)
            modalities[mod_name] = self.processor.preprocess_modality(
                mod_data, mod_name)

        # Convert all to tensors
        data_dict = {
            'eeg': {
                'time': torch.FloatTensor(eeg_features['time']),
                'freq': torch.FloatTensor(eeg_features['freq']['fft']),
                'connectivity': torch.FloatTensor(eeg_features['connectivity'])
            },
            'mod1': torch.FloatTensor(modalities['mod1']),
            'mod2': torch.FloatTensor(modalities['mod2']),
            'mod3': torch.FloatTensor(modalities['mod3'])
        }

        # Convert label
        label = 0 if file_info['label'] == 'A' else 1 if file_info['label'] == 'C' else 2

        return data_dict, label

def create_data_loaders(data_dir, batch_size=16):
    """Create train and test dataloaders"""
    # Load dataset information
    with open(os.path.join(data_dir, 'dataset_info.json'), 'r') as f:
        dataset_info = json.load(f)
    
    # Split data
    train_info = [d for d in dataset_info if d['split'] == 'train']
    test_info = [d for d in dataset_info if d['split'] == 'test']
    
    # Create processor
    processor = MultiModalProcessor()
    
    # Create datasets
    train_dataset = MultiModalDataset(data_dir, train_info, processor)
    test_dataset = MultiModalDataset(data_dir, test_info, processor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader