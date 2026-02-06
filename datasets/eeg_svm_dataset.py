import json
import os
import warnings

import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Enable CUDA
mne.utils.set_config("MNE_USE_CUDA", "true")
mne.cuda.init_cuda(verbose=False)  # Set to True for debugging


def load_eeg_data(file_path):
    """Load EEG data from EEGLAB file"""
    try:
        raw = mne.io.read_raw_eeglab(file_path)
        return raw.get_data()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


class EEGSVMDataset:
    """
    Dataset handler for EEG data to be used with SVM models.
    Loads and preprocesses EEG data from files.
    """

    def __init__(self, data_dir, data_info=None, scaler=None):
        """
        Initialize the dataset handler.

        Args:
            data_dir (str): Directory containing the EEG data files
            data_info (list): List of dictionaries containing file information
                             (if None, will try to load from labels.json)
            scaler (StandardScaler): Pre-fitted scaler for feature normalization
                                    (if None, a new one will be created and fitted)
        """
        self.data_dir = data_dir

        # Load data info if not provided
        if data_info is None:
            with open(os.path.join(data_dir, "labels.json"), "r") as f:
                self.data_info = json.load(f)
        else:
            self.data_info = data_info

        self.scaler = scaler

        # Generate data and labels lists like in the original EEGDataset
        self.data = [d["file_name"] for d in self.data_info]
        self.labels = [
            0 if d["label"] == "A" else 1 if d["label"] == "C" else 2
            for d in self.data_info
        ]

    def load_data(self, data_type=None):
        """
        Load and preprocess the EEG data.

        Args:
            data_type (str, optional): Type of data to load (e.g., 'train', 'test_cross', 'test_within')
                                      If None, all data will be loaded.

        Returns:
            tuple: (X, y) where X is the feature matrix and y contains the labels
        """
        X = []
        y = []

        # Filter by data type if specified
        if data_type is not None:
            filtered_info = [d for d in self.data_info if d["type"] == data_type]
        else:
            filtered_info = self.data_info

        print(f"Loading {len(filtered_info)} EEG samples...")

        # Load each EEG file
        for item in filtered_info:
            file_path = os.path.join(self.data_dir, item["file_name"])
            label = 0 if item["label"] == "A" else 1 if item["label"] == "C" else 2

            # Load the EEG data from the file
            eeg_data = load_eeg_data(file_path)

            if eeg_data is not None:
                # Preprocess EEG data for SVM
                features = self._preprocess_eeg(eeg_data)

                X.append(features)
                y.append(label)

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        if len(X) == 0:
            raise ValueError(
                "No valid data loaded. Please check file paths and data format."
            )

        # Create and fit scaler if not provided
        if self.scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, y

    def _preprocess_eeg(self, eeg_data):
        """
        Preprocess the EEG data for SVM input.

        Args:
            eeg_data (numpy.ndarray): Raw EEG data

        Returns:
            numpy.ndarray: Preprocessed features
        """
        # Extract meaningful features from EEG data
        # For SVM, we need to reduce dimensionality and extract relevant features

        # Assuming eeg_data shape is (channels, samples) or (samples, channels)
        # If data is (samples, channels), transpose it
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        num_channels = eeg_data.shape[0]

        # Feature extraction
        features = []

        # Time domain features
        for channel in range(num_channels):
            channel_data = eeg_data[channel, :]

            # Statistical features
            features.append(np.mean(channel_data))
            features.append(np.std(channel_data))
            features.append(np.max(channel_data))
            features.append(np.min(channel_data))
            features.append(
                np.percentile(channel_data, 75) - np.percentile(channel_data, 25)
            )

            # Energy
            features.append(np.sum(channel_data**2))

            # Zero crossings
            zero_crossings = np.where(np.diff(np.signbit(channel_data)))[0].shape[0]
            features.append(zero_crossings)

        # Frequency domain features (simplified)
        for channel in range(num_channels):
            channel_data = eeg_data[channel, :]
            fft_data = np.abs(np.fft.rfft(channel_data))

            # Frequency band powers
            n_freq = len(fft_data)

            # Delta (0-4 Hz)
            delta_range = max(1, int(n_freq * 4 / 128))  # Assuming 256 Hz sampling rate
            features.append(np.sum(fft_data[:delta_range]))

            # Theta (4-8 Hz)
            theta_range = max(1, int(n_freq * 8 / 128))
            features.append(np.sum(fft_data[delta_range:theta_range]))

            # Alpha (8-13 Hz)
            alpha_range = max(1, int(n_freq * 13 / 128))
            features.append(np.sum(fft_data[theta_range:alpha_range]))

            # Beta (13-30 Hz)
            beta_range = max(1, int(n_freq * 30 / 128))
            features.append(np.sum(fft_data[alpha_range:beta_range]))

            # Gamma (30+ Hz)
            features.append(np.sum(fft_data[beta_range:]))

        return np.array(features)

    def split_train_test(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.

        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X, y = self.load_data()
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def create_data_splits(self):
        """
        Create data splits for cross-validation.

        Returns:
            dict: Dictionary containing data splits
        """
        # Training data
        train_info = [d for d in self.data_info if d["type"] == "train"]
        X_train, y_train = self.load_data("train")

        # Cross-subject test data
        test_cross_info = [d for d in self.data_info if d["type"] == "test_cross"]
        X_test_cross, y_test_cross = self.load_data("test_cross")

        # Within-subject test data
        test_within_info = [d for d in self.data_info if d["type"] == "test_within"]
        X_test_within, y_test_within = self.load_data("test_within")

        return {
            "train": (X_train, y_train),
            "test_cross": (X_test_cross, y_test_cross),
            "test_within": (X_test_within, y_test_within),
        }


# Example usage
if __name__ == "__main__":
    data_dir = "model-data"

    # Create dataset
    dataset = EEGSVMDataset(data_dir)

    # Examine data structure
    with open(os.path.join(data_dir, "labels.json"), "r") as f:
        data_info = json.load(f)

    print("Sample data item structure:")
    print(data_info[0])

    # Load all data
    X, y = dataset.load_data()
    print(f"Loaded data shape: {X.shape}, labels shape: {y.shape}")
