import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from torch.utils.data import Dataset, DataLoader

class CWTPreprocessor:
    """
    Continuous Wavelet Transform preprocessor for EEG data.
    Transforms raw EEG signals into time-frequency representations using CWT.
    """
    def __init__(self, 
                 wavelet='morl',        # Morlet wavelet is good for EEG
                 num_scales=32,         # Number of scales for CWT
                 min_freq=1,            # Minimum frequency in Hz
                 max_freq=30,           # Maximum frequency in Hz (typical EEG range)
                 sampling_rate=250):    # Default EEG sampling rate
        self.wavelet = wavelet
        self.num_scales = num_scales
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sampling_rate = sampling_rate
        
        # Calculate scales based on frequencies
        self.scales = np.logspace(
            np.log10(self.max_freq / self.min_freq), 
            np.log10(1), 
            num=self.num_scales
        ) * self.sampling_rate / (2 * np.pi)
    
    def transform(self, eeg_data):
        """
        Apply CWT to EEG data.
        
        Args:
            eeg_data: numpy array of shape (channels, time_points)
            
        Returns:
            cwt_data: numpy array of shape (channels, num_scales, time_points)
        """
        num_channels, time_points = eeg_data.shape
        cwt_data = np.zeros((num_channels, self.num_scales, time_points))
        
        for ch in range(num_channels):
            channel_data = eeg_data[ch]
            
            # Apply CWT
            coeffs, _ = pywt.cwt(channel_data, self.scales, self.wavelet)
            
            # Store coefficients (power)
            cwt_data[ch] = np.abs(coeffs)**2
        
        return cwt_data


class CWTEEGDataset(Dataset):
    """
    Dataset for EEG data with CWT preprocessing.
    """
    def __init__(self, data_directory, dataset, scales=[3, 4, 5], 
                 wavelet='morl', num_scales=32, min_freq=1, max_freq=30):
        self.data_directory = data_directory
        self.dataset = dataset
        self.scales = scales
        self.labels = [d['label'] for d in dataset]
        self.data = [d['file_name'] for d in dataset]
        
        # Initialize CWT preprocessor
        self.cwt_processor = CWTPreprocessor(
            wavelet=wavelet,
            num_scales=num_scales,
            min_freq=min_freq,
            max_freq=max_freq
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        file_info = self.dataset[idx]
        file_path = os.path.join(self.data_directory, file_info['file_name'])
        
        # Load raw EEG data using your existing function (from previous code)
        eeg_data = load_eeg_data(file_path)
        
        # Apply CWT transform
        cwt_data = self.cwt_processor.transform(eeg_data)
        
        # Convert to tensor
        cwt_tensor = torch.FloatTensor(cwt_data)
        
        # Label (0 for 'A', 1 for 'C', 2 for 'F')
        label = 0 if file_info['label'] == 'A' else 1 if file_info['label'] == 'C' else 2
        
        return cwt_tensor, label


class TransformerEncoder(nn.Module):
    """
    Self-attention transformer encoder for CWT representations.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x shape: (seq_len, batch, dim)
        attended_x, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended_x))
        x = self.norm2(x + self.ffn(x))
        return x


class CWTTransformerEEG(nn.Module):
    """
    Transformer model for EEG classification using CWT representations.
    """
    def __init__(self, 
                 num_channels=19,        # Number of EEG channels
                 num_scales=32,          # Number of CWT scales
                 num_classes=3,          # Number of classes to predict
                 dim=128,                # Embedding dimension
                 num_layers=3,           # Number of transformer layers
                 num_heads=8,            # Number of attention heads
                 dropout=0.1):           # Dropout rate
        super().__init__()
        
        self.num_channels = num_channels
        self.num_scales = num_scales
        
        # Channel-wise embedding projection for CWT data
        self.channel_projection = nn.ModuleList([
            nn.Conv1d(
                in_channels=num_scales,  # Input: each channel's CWT scales
                out_channels=dim,        # Output: embedding dimension
                kernel_size=1            # 1x1 convolution (pointwise)
            ) for _ in range(num_channels)
        ])
        
        # Add frequency positional encoding
        self.freq_pos_encoding = nn.Parameter(
            torch.randn(num_channels, dim) * 0.02
        )
        
        # Transformer encoders
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass for CWT representations.
        
        Args:
            x: CWT coefficients of shape (batch, channels, scales, time)
            
        Returns:
            Classification output of shape (batch, num_classes)
        """
        batch_size, num_channels, num_scales, time_points = x.shape
        
        # Process each channel separately through its projection layer
        channel_embeddings = []
        
        for ch in range(num_channels):
            # Extract this channel's CWT representation
            ch_data = x[:, ch]  # (batch, scales, time)
            
            # Apply projection to get embedding
            ch_embedding = self.channel_projection[ch](ch_data)  # (batch, dim, time)
            
            # Average over time
            ch_embedding = ch_embedding.mean(dim=2)  # (batch, dim)
            
            channel_embeddings.append(ch_embedding)
        
        # Stack channel embeddings
        embeddings = torch.stack(channel_embeddings, dim=1)  # (batch, channels, dim)
        
        # Add positional encoding
        embeddings = embeddings + self.freq_pos_encoding.unsqueeze(0)  # (batch, channels, dim)
        
        # Prepare for transformer: (channels, batch, dim)
        embeddings = embeddings.permute(1, 0, 2)
        
        # Apply transformer layers
        for transformer in self.transformer_layers:
            embeddings = transformer(embeddings)
        
        # Global average pooling over channels
        embeddings = embeddings.mean(0)  # (batch, dim)
        
        # Apply classifier
        x = self.classifier[0](embeddings)  # LayerNorm
        x = self.classifier[1](x)  # Linear
        x = self.classifier[2](x)  # BatchNorm1d
        x = self.classifier[3](x)  # GELU
        x = self.classifier[4](x)  # Dropout
        x = self.classifier[5](x)  # Linear
        
        return x

class CWTTimeFrequencyEEG(nn.Module):
    """
    Alternative model that uses convolutional layers to process the CWT time-frequency representations
    before feeding into a transformer for sequence modeling.
    """
    def __init__(self, 
                 num_channels=19,        # Number of EEG channels
                 num_scales=32,          # Number of CWT scales
                 num_classes=3,          # Number of classes to predict
                 dim=128,                # Embedding dimension
                 num_layers=3,           # Number of transformer layers
                 num_heads=8,            # Number of attention heads
                 dropout=0.1):
        super().__init__()
        
        # Process each channel's time-frequency representation with a CNN
        self.time_freq_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            # Second convolutional layer
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            # Third convolutional layer
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Calculate the output size after CNN
        # This depends on input size and would need to be adjusted based on your data
        self.cnn_output_size = self._calculate_cnn_output_size(num_scales)
        
        # Project CNN output to transformer dimension
        self.projection = nn.Linear(self.cnn_output_size, dim)
        
        # Channel position encoding
        self.channel_pos_encoding = nn.Parameter(
            torch.randn(num_channels, dim) * 0.02
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )
    
    def _calculate_cnn_output_size(self, num_scales):
        """Calculate the output size of the CNN given the input scales"""
        # For each MaxPool2d operation, the size is reduced by factor of 2
        # We have 2 MaxPool2d layers
        scales_after_cnn = num_scales // 4
        time_after_cnn = num_scales // 4  # Assuming square input for simplicity
        return 64 * scales_after_cnn * time_after_cnn  # 64 is the number of output channels
    
    def forward(self, x):
        """
        Forward pass for CWT representations.
        
        Args:
            x: CWT coefficients of shape (batch, channels, scales, time)
            
        Returns:
            Classification output of shape (batch, num_classes)
        """
        batch_size, num_channels, num_scales, time_points = x.shape
        
        # Process each channel separately with CNN
        channel_features = []
        for ch in range(num_channels):
            # Extract channel data and add channel dimension for CNN
            ch_data = x[:, ch:ch+1, :, :]  # (batch, 1, scales, time)
            
            # Apply CNN
            ch_features = self.time_freq_cnn(ch_data)  # (batch, 64, scales/4, time/4)
            ch_features = ch_features.flatten(1)  # (batch, 64*scales/4*time/4)
            
            # Project to transformer dimension
            ch_features = self.projection(ch_features)  # (batch, dim)
            
            channel_features.append(ch_features)
        
        # Stack channel features
        features = torch.stack(channel_features, dim=1)  # (batch, channels, dim)
        
        # Add channel position encoding
        features = features + self.channel_pos_encoding.unsqueeze(0)  # (batch, channels, dim)
        
        # Prepare for transformer: (channels, batch, dim)
        features = features.permute(1, 0, 2)
        
        # Apply transformer layers
        for transformer in self.transformer_layers:
            features = transformer(features)
        
        # Global average pooling over channels
        features = features.mean(0)  # (batch, dim)
        
        # Apply classifier
        output = self.classifier(features)
        
        return output


def create_cwt_model(model_type='transformer', num_channels=19, num_classes=3, 
                     dim=128, num_scales=32, num_layers=3):
    """
    Factory function to create different types of CWT-based models.
    
    Args:
        model_type: 'transformer' or 'cnn_transformer'
        num_channels: Number of EEG channels
        num_classes: Number of classes to predict
        dim: Embedding dimension
        num_scales: Number of CWT scales
        num_layers: Number of transformer layers
        
    Returns:
        Model instance
    """
    if model_type == 'transformer':
        return CWTTransformerEEG(
            num_channels=num_channels,
            num_scales=num_scales,
            num_classes=num_classes,
            dim=dim,
            num_layers=num_layers
        )
    elif model_type == 'cnn_transformer':
        return CWTTimeFrequencyEEG(
            num_channels=num_channels,
            num_scales=num_scales,
            num_classes=num_classes,
            dim=dim,
            num_layers=num_layers
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage:
if __name__ == "__main__":
    # Parameters
    num_channels = 19
    num_scales = 32
    num_classes = 3
    dim = 128
    
    # Create sample input
    batch_size = 4
    time_points = 128
    x = torch.randn(batch_size, num_channels, num_scales, time_points)
    
    # Create model
    model = create_cwt_model('transformer', num_channels, num_classes, dim, num_scales)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")