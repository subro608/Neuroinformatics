import json
import os
import warnings

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Enable CUDA
mne.utils.set_config("MNE_USE_CUDA", "true")
mne.cuda.init_cuda(verbose=True)

# Define electrode positions in 10-20 system
ELECTRODE_POSITIONS = {
    "Fp1": {"x": -0.3, "y": 0.7, "lobe": "frontal", "hemisphere": "left"},
    "Fp2": {"x": 0.3, "y": 0.7, "lobe": "frontal", "hemisphere": "right"},
    "F7": {"x": -0.7, "y": 0.3, "lobe": "frontal", "hemisphere": "left"},
    "F3": {"x": -0.3, "y": 0.3, "lobe": "frontal", "hemisphere": "left"},
    "Fz": {"x": 0, "y": 0.3, "lobe": "frontal", "hemisphere": "central"},
    "F4": {"x": 0.3, "y": 0.3, "lobe": "frontal", "hemisphere": "right"},
    "F8": {"x": 0.7, "y": 0.3, "lobe": "frontal", "hemisphere": "right"},
    "T3": {"x": -0.7, "y": 0, "lobe": "temporal", "hemisphere": "left"},
    "C3": {"x": -0.3, "y": 0, "lobe": "central", "hemisphere": "left"},
    "Cz": {"x": 0, "y": 0, "lobe": "central", "hemisphere": "central"},
    "C4": {"x": 0.3, "y": 0, "lobe": "central", "hemisphere": "right"},
    "T4": {"x": 0.7, "y": 0, "lobe": "temporal", "hemisphere": "right"},
    "T5": {"x": -0.7, "y": -0.3, "lobe": "temporal", "hemisphere": "left"},
    "P3": {"x": -0.3, "y": -0.3, "lobe": "parietal", "hemisphere": "left"},
    "Pz": {"x": 0, "y": -0.3, "lobe": "parietal", "hemisphere": "central"},
    "P4": {"x": 0.3, "y": -0.3, "lobe": "parietal", "hemisphere": "right"},
    "T6": {"x": 0.7, "y": -0.3, "lobe": "temporal", "hemisphere": "right"},
    "O1": {"x": -0.3, "y": -0.7, "lobe": "occipital", "hemisphere": "left"},
    "O2": {"x": 0.3, "y": -0.7, "lobe": "occipital", "hemisphere": "right"},
}

# Encoding dictionaries
LOBE_ENCODING = {
    "frontal": 0,
    "temporal": 1,
    "central": 2,
    "parietal": 3,
    "occipital": 4,
}

HEMISPHERE_ENCODING = {"left": 0, "central": 1, "right": 2}


def load_eeg_data(file_path):
    """Load EEG data from EEGLAB file"""
    raw = mne.io.read_raw_eeglab(file_path)
    return raw.get_data()


def get_frequency_features(eeg_data):
    """Extract frequency domain features"""
    # Compute FFT
    fft_data = np.abs(np.fft.fft(eeg_data, axis=1))

    # Get power bands
    sampling_rate = 95  # As per the dataset
    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    band_powers = np.zeros((eeg_data.shape[0], len(freq_bands)))
    freqs = np.fft.fftfreq(eeg_data.shape[1], 1 / sampling_rate)

    for i, (band, (fmin, fmax)) in enumerate(freq_bands.items()):
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers[:, i] = np.mean(fft_data[:, band_mask], axis=1)

    return {"fft": fft_data, "band_powers": band_powers}


def get_spatial_features(eeg_data):
    """Extract spatial features based on electrode positions"""
    num_channels = eeg_data.shape[0]

    # Position matrix [x, y, lobe, hemisphere, signal_mean]
    pos_matrix = np.zeros((num_channels, 5))

    for idx, (channel, pos) in enumerate(ELECTRODE_POSITIONS.items()):
        pos_matrix[idx, 0] = pos["x"]
        pos_matrix[idx, 1] = pos["y"]
        pos_matrix[idx, 2] = LOBE_ENCODING[pos["lobe"]]
        pos_matrix[idx, 3] = HEMISPHERE_ENCODING[pos["hemisphere"]]
        pos_matrix[idx, 4] = np.mean(eeg_data[idx])

    # Distance matrix
    dist_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            dist_matrix[i, j] = np.sqrt(
                (pos_matrix[i, 0] - pos_matrix[j, 0]) ** 2
                + (pos_matrix[i, 1] - pos_matrix[j, 1]) ** 2
            )

    # Correlation matrix
    corr_matrix = np.corrcoef(eeg_data)

    return {
        "positions": pos_matrix,
        "distances": dist_matrix,
        "correlations": corr_matrix,
    }


class EEGDataset(Dataset):
    def __init__(self, data_directory, dataset, normalize=True):
        self.data_directory = data_directory
        self.dataset = dataset
        self.normalize = normalize
        self.labels = [d["label"] for d in dataset]
        self.data = [d["file_name"] for d in dataset]

    def __len__(self):
        return len(self.dataset)

    def normalize_data(self, data):
        """Z-score normalization"""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def __getitem__(self, idx):
        file_info = self.dataset[idx]
        file_path = os.path.join(self.data_directory, file_info["file_name"])

        # Load raw EEG data
        eeg_data = load_eeg_data(file_path)
        if self.normalize:
            eeg_data = self.normalize_data(eeg_data)
        eeg_data = eeg_data.astype("float32")

        # Get different views of the data
        freq_features = get_frequency_features(eeg_data)
        spatial_features = get_spatial_features(eeg_data)

        # Convert to tensors
        data_dict = {
            "time": torch.FloatTensor(eeg_data),
            "freq_fft": torch.FloatTensor(freq_features["fft"]),
            "freq_bands": torch.FloatTensor(freq_features["band_powers"]),
            "spatial_pos": torch.FloatTensor(spatial_features["positions"]),
            "spatial_dist": torch.FloatTensor(spatial_features["distances"]),
            "spatial_corr": torch.FloatTensor(spatial_features["correlations"]),
        }

        # Label
        label = (
            0 if file_info["label"] == "A" else 1 if file_info["label"] == "C" else 2
        )

        return data_dict, label


def create_data_loaders(data_dir, batch_size=16):
    """Create train and test dataloaders"""
    # Load data info
    with open(os.path.join(data_dir, "labels.json"), "r") as file:
        data_info = json.load(file)

    # Split data
    train_data = [d for d in data_info if d["type"] == "train"]
    test_data = [d for d in data_info if d["type"] == "test_cross"]

    # Create datasets
    train_dataset = EEGDataset(data_dir, train_data)
    test_dataset = EEGDataset(data_dir, test_data)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    data_dir = "model-data"
    train_loader, test_loader = create_data_loaders(data_dir, batch_size=16)

    # Print dataset information
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Show sample batch
    for batch, labels in train_loader:
        print("\nSample batch shapes:")
        for key, value in batch.items():
            print(f"{key}: {value.shape}")
        print(f"Labels shape: {labels.shape}")
        break
