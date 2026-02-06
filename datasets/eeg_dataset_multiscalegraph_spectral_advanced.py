import os
import pickle
import warnings

import mne
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import Dataset

# Suppress MNE and RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class SpatialSpectralEEGDataset(Dataset):
    """
    Enhanced dataset class for loading EEG signals with improved spatial information
    and spectral embeddings at multiple scales for the combined spatial-spectral model
    """

    def __init__(
        self,
        data_dir,
        data_info,
        scales=[3, 4, 5],
        time_length=8000,
        adjacency_type="combined",
    ):
        """
        Initialize the dataset

        Args:
            data_dir: Directory containing the data files
            data_info: List of dictionaries with file and label info
            scales: List of scales for the spectral embeddings
            time_length: Maximum time length for EEG signals (will be padded/truncated)
            adjacency_type: Type of adjacency matrix ('distance', 'correlation', or 'combined')
        """
        self.data_dir = data_dir
        self.data_info = data_info
        self.scales = scales
        self.time_length = time_length
        self.adjacency_type = adjacency_type

        # Extract file paths and labels
        self.data = [os.path.join(data_dir, info["file_name"]) for info in data_info]

        # Convert label strings to integers
        label_map = {"A": 0, "C": 1, "F": 2}
        self.labels = [label_map[info["label"]] for info in data_info]

        # Define spatial information - standard 10-20 electrode positions in 3D
        self.electrode_positions = {
            "Fp1": (-0.25, 0.9, 0.0),
            "Fp2": (0.25, 0.9, 0.0),
            "F7": (-0.7, 0.65, 0.0),
            "F3": (-0.4, 0.65, 0.0),
            "Fz": (0.0, 0.65, 0.0),
            "F4": (0.4, 0.65, 0.0),
            "F8": (0.7, 0.65, 0.0),
            "T3": (-0.85, 0.0, 0.0),
            "C3": (-0.4, 0.0, 0.1),
            "Cz": (0.0, 0.0, 0.2),
            "C4": (0.4, 0.0, 0.1),
            "T4": (0.85, 0.0, 0.0),
            "T5": (-0.7, -0.65, 0.0),
            "P3": (-0.4, -0.65, 0.0),
            "Pz": (0.0, -0.65, 0.1),
            "P4": (0.4, -0.65, 0.0),
            "T6": (0.7, -0.65, 0.0),
            "O1": (-0.25, -0.9, 0.0),
            "O2": (0.25, -0.9, 0.0),
        }

        # Create position matrix and normalize to [-1, 1]
        self.position_matrix = np.array(
            [pos for pos in self.electrode_positions.values()]
        )
        self.position_matrix = self.position_matrix / np.max(
            np.abs(self.position_matrix)
        )  # Normalize

        # Generate static adjacency matrix
        self.static_adjacency = self._create_adjacency_matrix(adjacency_type)

        # Store channel names
        self.channel_names = list(self.electrode_positions.keys())

        # Store lobe groups for functional connectivity analysis
        self.lobe_groups = {
            "frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
            "central": ["C3", "Cz", "C4"],
            "temporal": ["T3", "T4", "T5", "T6"],
            "parietal": ["P3", "Pz", "P4"],
            "occipital": ["O1", "O2"],
        }

        # Create lobe masks for faster access
        self.lobe_masks = self._create_lobe_masks()

        # Define frequency bands for spectral analysis
        self.frequency_bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 50),
        }

        print(
            f"Enhanced dataset created with {len(self.data)} samples across scales {scales}"
        )
        print(f"Using {adjacency_type} adjacency matrix")

    def _create_adjacency_matrix(self, adjacency_type):
        """Create static adjacency matrix based on specified type"""
        num_channels = len(self.electrode_positions)

        if adjacency_type == "distance" or adjacency_type == "combined":
            distances = squareform(pdist(self.position_matrix))
            adjacency = 1.0 / (1.0 + distances)
            np.fill_diagonal(adjacency, 1.0)
        else:  # Default to identity for correlation-only (dynamic will override)
            adjacency = np.eye(num_channels)

        return adjacency / adjacency.max()

    def _create_lobe_masks(self):
        """Create binary masks for each lobe"""
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
        """Compute dynamic adjacency matrix based on EEG data"""
        num_channels = eeg_data.shape[0]
        correlation_matrix = np.corrcoef(eeg_data)
        correlation_matrix = np.nan_to_num(correlation_matrix, 0)  # Replace NaN with 0
        correlation_matrix = np.abs(correlation_matrix)  # Ensure positive

        if self.adjacency_type == "correlation":
            adjacency = correlation_matrix
        elif self.adjacency_type == "combined":
            adjacency = 0.5 * self.static_adjacency + 0.5 * correlation_matrix
        else:  # 'distance'
            adjacency = self.static_adjacency

        return adjacency / adjacency.max()

    def compute_spectral_embeddings(self, eeg_data, sfreq=250):
        """Compute enhanced spectral embeddings with per-lobe features"""
        num_channels = eeg_data.shape[0]
        spectral_embeddings = {}

        scale_freqs = {
            3: [(0.5, 8), (8, 13), (13, 30)],
            4: [(0.5, 4), (4, 8), (8, 13), (13, 30)],
            5: [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)],
            6: [(0.5, 2), (2, 4), (4, 8), (8, 13), (13, 30), (30, 50)],
        }

        for scale in self.scales:
            freqs = scale_freqs.get(scale, list(self.frequency_bands.values())[:scale])
            embeddings = np.zeros((num_channels, len(freqs)))
            lobe_embeddings = {lobe: np.zeros(len(freqs)) for lobe in self.lobe_groups}

            for ch_idx in range(num_channels):
                ch_data = eeg_data[ch_idx]
                if np.all(ch_data == 0):
                    continue

                f, psd = signal.welch(
                    ch_data,
                    fs=sfreq,
                    nperseg=min(512, len(ch_data) // 4),
                    noverlap=min(256, len(ch_data) // 8),
                )
                for band_idx, (fmin, fmax) in enumerate(freqs):
                    idx = np.logical_and(f >= fmin, f <= fmax)
                    if np.any(idx):
                        embeddings[ch_idx, band_idx] = np.mean(psd[idx])

                # Update lobe averages
                for lobe, mask in self.lobe_masks.items():
                    if mask[ch_idx]:
                        lobe_embeddings[lobe] += embeddings[ch_idx] / mask.sum()

            embeddings = np.log(embeddings + 1e-10)  # Log transform
            for lobe in lobe_embeddings:
                lobe_embeddings[lobe] = np.log(lobe_embeddings[lobe] + 1e-10)

            # Combine channel and lobe embeddings
            combined_embeddings = np.vstack(
                [embeddings] + [lobe_embeddings[lobe] for lobe in self.lobe_groups]
            )
            spectral_embeddings[f"scale_{scale}"] = combined_embeddings

        return spectral_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a sample from the dataset with enhanced features"""
        file_path = self.data[idx]
        label = self.labels[idx]

        raw_eeg = None
        spectral_embeddings = {}
        dynamic_adjacency = None

        try:
            if file_path.endswith(".pkl"):
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                if "eeg" in data:
                    raw_eeg = data["eeg"]
                for scale in self.scales:
                    scale_key = f"scale_{scale}"
                    if scale_key in data:
                        spectral_embeddings[scale_key] = torch.FloatTensor(
                            data[scale_key]
                        )

            elif file_path.endswith(".set"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
                eeg_data = raw.get_data()

                if len(eeg_data) >= 19:
                    if set(self.channel_names).issubset(set(raw.ch_names)):
                        ch_indices = [
                            raw.ch_names.index(ch)
                            for ch in self.channel_names
                            if ch in raw.ch_names
                        ]
                        eeg_data = eeg_data[ch_indices]
                    else:
                        eeg_data = eeg_data[:19]
                if len(eeg_data) < 19:
                    padding = np.zeros((19 - len(eeg_data), eeg_data.shape[1]))
                    eeg_data = np.vstack([eeg_data, padding])

                raw_eeg = eeg_data
                dynamic_adjacency = self._compute_dynamic_adjacency(eeg_data)
                spectral_embeddings = self.compute_spectral_embeddings(
                    eeg_data, sfreq=raw.info["sfreq"]
                )
                for scale_key in spectral_embeddings:
                    spectral_embeddings[scale_key] = torch.FloatTensor(
                        spectral_embeddings[scale_key]
                    )

            if raw_eeg is None:
                raw_eeg_path = file_path.replace(
                    os.path.splitext(file_path)[1], "_raw.npy"
                )
                if os.path.exists(raw_eeg_path):
                    raw_eeg = np.load(raw_eeg_path)
                else:
                    raw_eeg = np.zeros((19, self.time_length))

            for scale in self.scales:
                scale_key = f"scale_{scale}"
                if scale_key not in spectral_embeddings:
                    scale_path = file_path.replace(
                        os.path.splitext(file_path)[1], f"_{scale}.npy"
                    )
                    if os.path.exists(scale_path):
                        spectral_embeddings[scale_key] = torch.FloatTensor(
                            np.load(scale_path)
                        )
                    elif not np.all(raw_eeg == 0):
                        if not spectral_embeddings:
                            spectral_embeddings = self.compute_spectral_embeddings(
                                raw_eeg
                            )
                        spectral_embeddings[scale_key] = torch.FloatTensor(
                            spectral_embeddings[scale_key]
                        )
                    else:
                        spectral_embeddings[scale_key] = torch.zeros(
                            19 + len(self.lobe_groups), scale
                        )

            # Process raw EEG
            if raw_eeg.shape[1] > self.time_length:
                raw_eeg = raw_eeg[:, : self.time_length]
            elif raw_eeg.shape[1] < self.time_length:
                padding = np.zeros(
                    (raw_eeg.shape[0], self.time_length - raw_eeg.shape[1])
                )
                raw_eeg = np.concatenate([raw_eeg, padding], axis=1)

            if not np.all(raw_eeg == 0):
                std = raw_eeg.std(axis=1, keepdims=True)
                std[std < 1e-6] = 1.0
                raw_eeg = (raw_eeg - raw_eeg.mean(axis=1, keepdims=True)) / std

            for scale_key in spectral_embeddings:
                embed = spectral_embeddings[scale_key].numpy()
                if not np.all(embed == 0):
                    std = embed.std(axis=1, keepdims=True)
                    std[std < 1e-6] = 1.0
                    embed = (embed - embed.mean(axis=1, keepdims=True)) / std
                    spectral_embeddings[scale_key] = torch.FloatTensor(embed)

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raw_eeg = np.zeros((19, self.time_length))
            spectral_embeddings = {
                f"scale_{scale}": torch.zeros(19 + len(self.lobe_groups), scale)
                for scale in self.scales
            }
            dynamic_adjacency = self.static_adjacency

        sample = {
            "raw_eeg": torch.FloatTensor(raw_eeg),
            "adjacency": torch.FloatTensor(
                dynamic_adjacency
                if dynamic_adjacency is not None
                else self.static_adjacency
            ),
            "spatial_positions": torch.FloatTensor(self.position_matrix),
        }
        sample.update(spectral_embeddings)

        return sample, label

    def get_adjacency(self):
        """Return the static adjacency matrix"""
        return self.static_adjacency

    def get_electrode_positions(self):
        """Return electrode positions"""
        return self.electrode_positions

    def get_lobe_masks(self):
        """Return lobe masks"""
        return self.lobe_masks


def create_dataset(
    data_dir, data_info, scales=[3, 4, 5], time_length=8000, adjacency_type="combined"
):
    """Helper function to create the dataset"""
    return SpatialSpectralEEGDataset(
        data_dir=data_dir,
        data_info=data_info,
        scales=scales,
        time_length=time_length,
        adjacency_type=adjacency_type,
    )
