import json
import os
import warnings

import mne
import numpy as np
import pywt
import torch
from torch.utils.data import DataLoader, Dataset

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

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


def get_wavelet_features(eeg_data):
    """Extract wavelet transform features using DWT and CWT"""
    num_channels, signal_length = eeg_data.shape
    sampling_rate = 95  # As per the dataset

    # Discrete Wavelet Transform (DWT)
    wavelet = "db4"  # Daubechies wavelet (good for EEG)
    level = 5  # Decomposition level

    # Initialize result arrays
    dwt_coeffs = []
    wavelet_entropy = np.zeros((num_channels,))
    wavelet_energy = np.zeros(
        (num_channels, level + 1)
    )  # +1 for approximation coefficients

    for ch in range(num_channels):
        # Apply DWT
        coeffs = pywt.wavedec(eeg_data[ch], wavelet, level=level)

        # Calculate wavelet entropy
        norm_coeffs = [
            c / np.sqrt(np.sum(c**2)) if np.sum(c**2) > 0 else c for c in coeffs
        ]
        for c in norm_coeffs:
            p = c**2
            p = p[p > 0]  # Avoid log(0)
            if len(p) > 0:
                wavelet_entropy[ch] -= np.sum(p * np.log(p))

        # Calculate wavelet energy per band
        for i, c in enumerate(coeffs):
            wavelet_energy[ch, i] = np.sum(c**2)

        # Store coefficients (could be large, using only approximation and first detail)
        dwt_coeffs.append(np.concatenate([coeffs[0], coeffs[1]]))

    # Continuous Wavelet Transform (CWT) - using Morlet wavelet
    # We'll compute CWT for a few specific scales to capture different frequency bands
    scales = np.arange(1, 64)  # Scales corresponding roughly to 0.5-30Hz

    # Store CWT coefficients for key scales only (to manage memory)
    key_scales = [1, 4, 8, 16, 32, 48]  # Corresponding to different frequency bands
    cwt_features = np.zeros((num_channels, len(key_scales), min(256, signal_length)))

    for ch in range(num_channels):
        # Compute CWT
        coef, freqs = pywt.cwt(
            eeg_data[ch], scales, "morl", sampling_period=1 / sampling_rate
        )

        # Store key scales
        for i, scale in enumerate(key_scales):
            if scale < len(scales):
                # Take magnitude and subsample if needed
                cwt_mag = np.abs(coef[scale])
                if len(cwt_mag) > 256:
                    # Subsample to manage memory
                    idx = np.linspace(0, len(cwt_mag) - 1, 256).astype(int)
                    cwt_features[ch, i] = cwt_mag[idx]
                else:
                    cwt_features[ch, i, : len(cwt_mag)] = cwt_mag

    # Calculate normalized time-frequency energy distribution
    cwt_energy = np.sum(cwt_features**2, axis=2)
    total_energy = np.sum(cwt_energy, axis=1, keepdims=True)
    cwt_energy_norm = cwt_energy / (total_energy + 1e-10)

    return {
        "dwt_coeffs": np.array(dwt_coeffs),
        "wavelet_entropy": wavelet_entropy,
        "wavelet_energy": wavelet_energy,
        "cwt_features": cwt_features,
        "cwt_energy_norm": cwt_energy_norm,
    }


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
    def __init__(self, data_directory, dataset, normalize=True, include_wavelet=True):
        self.data_directory = data_directory
        self.dataset = dataset
        self.normalize = normalize
        self.include_wavelet = include_wavelet
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

        # Create data dictionary
        data_dict = {
            "time": torch.FloatTensor(eeg_data),
            "freq_fft": torch.FloatTensor(freq_features["fft"]),
            "freq_bands": torch.FloatTensor(freq_features["band_powers"]),
            "spatial_pos": torch.FloatTensor(spatial_features["positions"]),
            "spatial_dist": torch.FloatTensor(spatial_features["distances"]),
            "spatial_corr": torch.FloatTensor(spatial_features["correlations"]),
        }

        # Add wavelet features if requested
        if self.include_wavelet:
            try:
                wavelet_features = get_wavelet_features(eeg_data)
                data_dict.update(
                    {
                        "wavelet_dwt": torch.FloatTensor(
                            wavelet_features["dwt_coeffs"]
                        ),
                        "wavelet_entropy": torch.FloatTensor(
                            wavelet_features["wavelet_entropy"]
                        ),
                        "wavelet_energy": torch.FloatTensor(
                            wavelet_features["wavelet_energy"]
                        ),
                        "wavelet_cwt_energy": torch.FloatTensor(
                            wavelet_features["cwt_energy_norm"]
                        ),
                    }
                )

                # Only include CWT features if they're not too large
                if (
                    wavelet_features["cwt_features"].size < 10e6
                ):  # Size check to avoid memory issues
                    data_dict["wavelet_cwt"] = torch.FloatTensor(
                        wavelet_features["cwt_features"]
                    )
            except Exception as e:
                print(
                    f"Warning: Could not compute wavelet features for {file_info['file_name']}: {e}"
                )
                # Provide empty tensors as fallback
                num_channels = eeg_data.shape[0]
                data_dict.update(
                    {
                        "wavelet_dwt": torch.zeros(
                            (num_channels, 128), dtype=torch.float32
                        ),
                        "wavelet_entropy": torch.zeros(
                            (num_channels,), dtype=torch.float32
                        ),
                        "wavelet_energy": torch.zeros(
                            (num_channels, 6), dtype=torch.float32
                        ),
                        "wavelet_cwt_energy": torch.zeros(
                            (num_channels, 6), dtype=torch.float32
                        ),
                    }
                )

        # Label
        label = (
            0 if file_info["label"] == "A" else 1 if file_info["label"] == "C" else 2
        )

        return data_dict, label


def create_data_loaders(data_dir, batch_size=16, include_wavelet=True):
    """Create train and test dataloaders"""
    # Load data info
    with open(os.path.join(data_dir, "labels.json"), "r") as file:
        data_info = json.load(file)

    # Split data
    train_data = [d for d in data_info if d["type"] == "train"]
    test_data = [d for d in data_info if d["type"] == "test_cross"]

    # Create datasets
    train_dataset = EEGDataset(data_dir, train_data, include_wavelet=include_wavelet)
    test_dataset = EEGDataset(data_dir, test_data, include_wavelet=include_wavelet)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    data_dir = "model-data"
    train_loader, test_loader = create_data_loaders(
        data_dir, batch_size=16, include_wavelet=True
    )

    # Print dataset information
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Show sample batch
    for batch, labels in train_loader:
        print("\nSample batch shapes:")
        for key, value in batch.items():
            print(f"{key}: {value.shape}")
        print(f"Labels shape: {labels.shape}")

        # Print specific information about wavelet features
        if "wavelet_dwt" in batch:
            print("\nWavelet feature information:")
            print(f"DWT coefficients: {batch['wavelet_dwt'].shape}")
            print(f"Wavelet entropy: {batch['wavelet_entropy'].shape}")
            print(f"Wavelet energy bands: {batch['wavelet_energy'].shape}")
            if "wavelet_cwt" in batch:
                print(f"CWT features: {batch['wavelet_cwt'].shape}")
            print(f"CWT energy distribution: {batch['wavelet_cwt_energy'].shape}")

        break
