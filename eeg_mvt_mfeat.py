import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

class MVTEncoder(nn.Module):
    def __init__(self, dim, num_heads=16):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(2 * dim, dim)
        )
        
    def forward(self, x):
        x = self.pre_norm(x)
        att_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(att_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class CrossViewAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        q = self.pre_norm(q)
        att_out, _ = self.attention(q, k, v)
        return self.norm(q + self.dropout(att_out))

class MVTEEG(nn.Module):
    def __init__(self, num_channels=19, num_classes=3, dim=128):
        super().__init__()
        
        # Time domain processing
        self.time_conv = nn.Sequential(
            nn.Conv1d(num_channels, dim, kernel_size=64, stride=32),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.time_encoder = MVTEncoder(dim)
        
        # Frequency domain processing
        self.freq_conv = nn.Sequential(
            nn.Conv1d(num_channels, dim, kernel_size=32, stride=16),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.freq_encoder = MVTEncoder(dim)
        
        # NEW: Wavelet transform processing
        self.wavelet_conv = nn.Sequential(
            nn.Conv1d(num_channels, dim, kernel_size=32, stride=16),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.wavelet_encoder = MVTEncoder(dim)
        
        # Spatial processing
        self.spatial_embed = nn.Sequential(
            nn.Linear(num_channels, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.spatial_encoder = MVTEncoder(dim)
        
        # Cross-view attention - updated with wavelet domain
        self.time_freq_attention = CrossViewAttention(dim)
        self.time_wavelet_attention = CrossViewAttention(dim)  # New
        self.time_spatial_attention = CrossViewAttention(dim)
        self.freq_wavelet_attention = CrossViewAttention(dim)  # New
        self.freq_spatial_attention = CrossViewAttention(dim)
        self.wavelet_spatial_attention = CrossViewAttention(dim)  # New
        
        # Fusion - updated for 4 views instead of 3
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 4),  # Changed from dim * 3
            nn.Linear(dim * 4, dim),  # Changed from dim * 3
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(dim)
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, num_classes)
        )

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.apply(_init_weights)
    
    def get_frequency_features(self, x):
        # Apply FFT and scale
        x_fft = torch.fft.fft(x, dim=-1).abs()
        x_fft = torch.log1p(x_fft)  # Log scaling
        # Normalize
        x_fft = (x_fft - x_fft.mean(dim=-1, keepdim=True)) / (x_fft.std(dim=-1, keepdim=True) + 1e-8)
        return x_fft
    
    def get_wavelet_features(self, x):
        # Move to CPU for wavelet transform (PyWavelets doesn't support GPU)
        device = x.device
        x_cpu = x.detach().cpu().numpy()
        
        batch_size, channels, signal_length = x_cpu.shape
        
        # Define wavelet parameters
        wavelet = 'db4'  # Daubechies wavelet with 4 vanishing moments
        level = 5  # Decomposition level
        
        # Apply CWT or DWT to each channel in each batch
        # For simplicity, we'll use DWT and convert coefficients to features
        coeffs_batch = []
        
        for b in range(batch_size):
            coeffs_channels = []
            for c in range(channels):
                # Decompose signal using DWT
                coeffs = pywt.wavedec(x_cpu[b, c], wavelet, level=level)
                
                # Concatenate coefficients (could also do something more sophisticated here)
                # This approach preserves some time-frequency information
                features = np.concatenate([c for c in coeffs])
                
                # Pad or truncate to a fixed length to ensure consistent size
                target_length = signal_length
                if len(features) >= target_length:
                    features = features[:target_length]
                else:
                    features = np.pad(features, (0, target_length - len(features)))
                
                coeffs_channels.append(features)
            
            coeffs_batch.append(np.stack(coeffs_channels))
        
        # Convert back to tensor
        wavelet_features = torch.tensor(np.stack(coeffs_batch), dtype=torch.float32).to(device)
        
        # Normalize features
        wavelet_features = (wavelet_features - wavelet_features.mean(dim=-1, keepdim=True)) / (wavelet_features.std(dim=-1, keepdim=True) + 1e-8)
        
        return wavelet_features
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size = x.shape[0]
        
        # Time domain
        time_features = self.time_conv(x)
        time_features = time_features.permute(2, 0, 1)  # (seq, batch, dim)
        time_features = self.time_encoder(time_features)
        
        # Frequency domain
        freq_features = self.get_frequency_features(x)
        freq_features = self.freq_conv(freq_features)
        freq_features = freq_features.permute(2, 0, 1)
        freq_features = self.freq_encoder(freq_features)
        
        # Wavelet domain - new feature
        wavelet_features = self.get_wavelet_features(x)
        wavelet_features = self.wavelet_conv(wavelet_features)
        wavelet_features = wavelet_features.permute(2, 0, 1)
        wavelet_features = self.wavelet_encoder(wavelet_features)
        
        # Spatial domain
        spatial_features = self.spatial_embed(x.permute(0, 2, 1))  # (batch, time, dim)
        spatial_features = spatial_features.permute(1, 0, 2)
        spatial_features = self.spatial_encoder(spatial_features)
        
        # Cross-view attention with gradient scaling
        with torch.cuda.amp.autocast(enabled=True):
            # Existing cross-attention
            time_freq = self.time_freq_attention(time_features, freq_features, freq_features)
            time_spatial = self.time_spatial_attention(time_features, spatial_features, spatial_features)
            freq_spatial = self.freq_spatial_attention(freq_features, spatial_features, spatial_features)
            
            # New cross-attention with wavelet
            time_wavelet = self.time_wavelet_attention(time_features, wavelet_features, wavelet_features)
            freq_wavelet = self.freq_wavelet_attention(freq_features, wavelet_features, wavelet_features)
            wavelet_spatial = self.wavelet_spatial_attention(wavelet_features, spatial_features, spatial_features)
        
        # Fusion with scaling - updated to include wavelet features
        time_pool = time_features.mean(0)  # Original pooling
        freq_pool = freq_features.mean(0)  # Original pooling
        wavelet_pool = wavelet_features.mean(0)  # New pooling
        spatial_pool = spatial_features.mean(0)  # Original pooling
        
        combined = torch.cat([time_pool, freq_pool, wavelet_pool, spatial_pool], dim=-1)
        fused = self.fusion(combined)
        
        # Classification
        output = self.classifier(fused)
        
        return output

# Initialize model with proper weight initialization
def create_model(num_channels=19, num_classes=3, dim=128):
    model = MVTEEG(num_channels, num_classes, dim)
    model.init_weights()
    return model