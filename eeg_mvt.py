import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
       
       # Spatial processing
       self.spatial_embed = nn.Sequential(
           nn.Linear(num_channels, dim),
           nn.LayerNorm(dim),
           nn.GELU(),
           nn.Dropout(0.1)
       )
       self.spatial_encoder = MVTEncoder(dim)
       
       # Cross-view attention
       self.time_freq_attention = CrossViewAttention(dim)
       self.time_spatial_attention = CrossViewAttention(dim)
       self.freq_spatial_attention = CrossViewAttention(dim)
       
       # Fusion
       self.fusion = nn.Sequential(
           nn.LayerNorm(dim * 3),
           nn.Linear(dim * 3, dim),
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
       
       # Spatial domain
       spatial_features = self.spatial_embed(x.permute(0, 2, 1))  # (batch, time, dim)
       spatial_features = spatial_features.permute(1, 0, 2)
       spatial_features = self.spatial_encoder(spatial_features)
       
       # Cross-view attention with gradient scaling
       with torch.cuda.amp.autocast(enabled=True):
           time_freq = self.time_freq_attention(time_features, freq_features, freq_features)
           time_spatial = self.time_spatial_attention(time_features, spatial_features, spatial_features)
           freq_spatial = self.freq_spatial_attention(freq_features, spatial_features, spatial_features)
       
       # Fusion with scaling
       time_pool = time_freq.mean(0)
       freq_pool = freq_spatial.mean(0)
       spatial_pool = time_spatial.mean(0)
       
       combined = torch.cat([time_pool, freq_pool, spatial_pool], dim=-1)
       fused = self.fusion(combined)
       
       # Classification
       output = self.classifier(fused)
       
       return output

# Initialize model with proper weight initialization
def create_model(num_channels=19, num_classes=3, dim=128):
   model = MVTEEG(num_channels, num_classes, dim)
   model.init_weights()
   return model