import os
import mne
import warnings
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import scipy.sparse.linalg as slinalg

# Ignore RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_eeg_data(file_path):
    """Load EEG data from EEGLAB file"""
    raw = mne.io.read_raw_eeglab(file_path)
    return raw.get_data()

def compute_spectral_embeddings(eeg_data, scales=[3, 4, 5]):
    """Extract spectral clustering embeddings from EEG data at multiple scales"""
    # Compute coherence/correlation matrix
    coherence = np.corrcoef(eeg_data)
    
    # Create adjacency matrix (absolute value of coherence)
    adjacency = np.abs(coherence)
    np.fill_diagonal(adjacency, 0)  # Zero out diagonal
    
    # Compute the normalized Laplacian
    degree = np.sum(adjacency, axis=1)
    degree_sqrt_inv = np.diag(1.0 / np.sqrt(degree + 1e-8))
    laplacian = np.eye(eeg_data.shape[0]) - degree_sqrt_inv @ adjacency @ degree_sqrt_inv
    
    # Compute eigenvectors for each scale
    embeddings_dict = {}
    
    for scale in scales:
        try:
            # Try sparse eigendecomposition first (faster for large matrices)
            eigenvalues, eigenvectors = slinalg.eigsh(laplacian, k=scale+1, which='SM')
            
            # Use k smallest non-zero eigenvalues (first is often zero)
            idx = np.argsort(eigenvalues)[1:scale+1]
            embedding = eigenvectors[:, idx]
            
        except:
            # Fallback to full eigendecomposition if sparse fails
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            
            # Use k smallest non-zero eigenvalues
            idx = np.argsort(eigenvalues)[1:scale+1]
            embedding = eigenvectors[:, idx]
        
        embeddings_dict[f'scale_{scale}'] = embedding
    
    return embeddings_dict, adjacency

class MultiScaleSpectralEEGDataset(Dataset):
    def __init__(self, data_directory, dataset, normalize=True, scales=[3, 4, 5]):
        self.data_directory = data_directory
        self.dataset = dataset
        self.normalize = normalize
        self.scales = scales
        self.labels = [d['label'] for d in dataset]
        self.data = [d['file_name'] for d in dataset]

    def __len__(self):
        return len(self.dataset)

    def normalize_data(self, data):
        """Z-score normalization"""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def __getitem__(self, idx):
        file_info = self.dataset[idx]
        file_path = os.path.join(self.data_directory, file_info['file_name'])

        # Load raw EEG data
        eeg_data = load_eeg_data(file_path)
        if self.normalize:
            eeg_data = self.normalize_data(eeg_data)
        eeg_data = eeg_data.astype('float32')

        # Compute spectral clustering embeddings at multiple scales
        spectral_dict, adjacency = compute_spectral_embeddings(eeg_data, self.scales)
        
        # Convert to tensors
        result_dict = {'eeg': torch.FloatTensor(eeg_data),
                        'adjacency': torch.FloatTensor(adjacency)}
        
        for scale in self.scales:
            result_dict[f'scale_{scale}'] = torch.FloatTensor(spectral_dict[f'scale_{scale}'])
        
        # Label (0 for 'A', 1 for 'C', 2 for 'F')
        # For Alzheimer's binary classification, you might use:
        # label = 1 if file_info['label'] == 'A' else 0
        label = 0 if file_info['label'] == 'A' else 1 if file_info['label'] == 'C' else 2

        return result_dict, label

def create_data_loaders(data_dir, batch_size=16, scales=[3, 4, 5]):
    """Create train and test dataloaders"""
    # Load data info
    with open(os.path.join(data_dir, 'labels.json'), 'r') as file:
        data_info = json.load(file)
    
    # Split data
    train_data = [d for d in data_info if d['type'] == 'train']
    test_data = [d for d in data_info if d['type'] == 'test_cross']
    
    # Create datasets
    train_dataset = MultiScaleSpectralEEGDataset(data_dir, train_data, scales=scales)
    test_dataset = MultiScaleSpectralEEGDataset(data_dir, test_data, scales=scales)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    data_dir = "model-data"
    scales = [3, 4, 5]
    train_loader, test_loader = create_data_loaders(data_dir, batch_size=16, scales=scales)
    
    # Print dataset information
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Show sample batch
    for batch, labels in train_loader:
        print("\nSample batch shapes:")
        print(f"EEG data: {batch['eeg'].shape}")
        print(f"Adjacency: {batch['adjacency'].shape}")
        
        for scale in scales:
            print(f"Spectral embeddings (scale={scale}): {batch[f'scale_{scale}'].shape}")
            
        print(f"Labels shape: {labels.shape}")
        break