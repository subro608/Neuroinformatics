import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MVTEncoder(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
        # Added batch normalization after attention
        self.bn1 = nn.BatchNorm1d(dim)
        
        # Added batch normalization in FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.BatchNorm1d(2 * dim),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(2 * dim, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x):
        # x shape: (seq_len, batch, dim)
        x = self.pre_norm(x)
        att_out, _ = self.attention(x, x, x)
        
        # Handle BatchNorm1d dimensions
        batch_size = x.shape[1]
        att_out_bn = att_out.permute(1, 2, 0)  # (batch, dim, seq_len)
        att_out_bn = self.bn1(att_out_bn).permute(2, 0, 1)  # Back to (seq_len, batch, dim)
        
        x = self.norm1(x + self.dropout(att_out_bn))
        
        # Handle FFN with BatchNorm1d
        x_ffn = x.permute(1, 0, 2)  # (batch, seq_len, dim)
        
        # Linear
        x_ffn = self.ffn[0](x_ffn)  # (batch, seq_len, 2*dim)
        
        # BatchNorm1d
        x_ffn = x_ffn.permute(0, 2, 1)  # (batch, 2*dim, seq_len)
        x_ffn = self.ffn[1](x_ffn)
        x_ffn = x_ffn.permute(0, 2, 1)  # (batch, seq_len, 2*dim)
        
        # GELU and Dropout
        x_ffn = self.ffn[2](x_ffn)  
        x_ffn = self.ffn[3](x_ffn)
        
        # Second Linear
        x_ffn = self.ffn[4](x_ffn)  # (batch, seq_len, dim)
        
        # Final BatchNorm1d
        x_ffn = x_ffn.permute(0, 2, 1)  # (batch, dim, seq_len)
        x_ffn = self.ffn[5](x_ffn)
        x_ffn = x_ffn.permute(0, 2, 1)  # (batch, seq_len, dim)
        
        # Convert back to (seq_len, batch, dim)
        x_ffn = x_ffn.permute(1, 0, 2)
        
        x = self.norm2(x + self.dropout(x_ffn))
        return x

class SpectralTransformerEEG(nn.Module):
    def __init__(self, num_channels=19, num_classes=3, dim=128, num_clusters=5, num_layers=3):
        super().__init__()
        self.spectral_proj = nn.Linear(num_clusters, dim)
        
        # Added input batch normalization
        self.bn_input = nn.BatchNorm1d(dim)
        
        # Position encoding for electrode positions
        self.pos_encoding = nn.Parameter(torch.randn(num_channels, dim) * 0.02)
        
        # Stack of transformer encoders
        self.encoder_layers = nn.ModuleList([
            MVTEncoder(dim, num_heads=8) for _ in range(num_layers)
        ])
        
        # Classification head with batch normalization
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, num_classes)
        )

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.apply(_init_weights)
    
    def forward(self, x):
        # x is expected to be spectral embeddings of shape (batch, channels, clusters)
        batch_size, num_channels, _ = x.shape
        
        # Project spectral embeddings to the model dimension
        x = self.spectral_proj(x)  # (batch, channels, dim)
        
        # Apply batch normalization
        x_bn = x.permute(0, 2, 1)  # (batch, dim, channels)
        x_bn = self.bn_input(x_bn)
        x = x_bn.permute(0, 2, 1)  # (batch, channels, dim)
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0)  # (batch, channels, dim)
        
        # Transpose for transformer input (seq_len, batch, dim)
        x = x.permute(1, 0, 2)  # (channels, batch, dim)
        
        # Apply transformer encoder layers
        for encoder in self.encoder_layers:
            x = encoder(x)
        
        # Global average pooling over channels
        x = x.mean(0)  # (batch, dim)
        
        # Classification - handle batch norm dimensions
        x = self.classifier[0](x)  # LayerNorm
        x = self.classifier[1](x)  # Linear
        x = self.classifier[2](x)  # BatchNorm1d
        x = self.classifier[3](x)  # GELU
        x = self.classifier[4](x)  # Dropout
        output = self.classifier[5](x)  # Linear
        
        return output

# Keep the SpectralGraphTransformer class

class SpectralGraphTransformer(nn.Module):
    def __init__(self, num_channels=19, num_classes=3, dim=256, num_clusters=5, num_layers=4):
        super().__init__()
        
        # Increase initial dimensionality
        self.embed_dim = dim * 2  # Double the dimension for more capacity
        
        # Embedding of spectral features with increased capacity
        self.embed = nn.Sequential(
            nn.Linear(num_clusters, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim)
        )
        
        # Increase number of heads for more expressive attention
        num_heads = 8  # Increased from 4
        
        # Deeper graph attention structure
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # Multi-head attention with more heads
            self.gat_layers.append(nn.MultiheadAttention(self.embed_dim, num_heads=num_heads, dropout=0.1))
            self.layer_norms.append(nn.LayerNorm(self.embed_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.embed_dim))
            
            # Add FFN after each attention layer (similar to transformer blocks)
            ffn = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 4),  # Wider FFN
                nn.BatchNorm1d(self.embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.embed_dim * 4, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim)
            )
            self.ffn_layers.append(ffn)
            self.ffn_norms.append(nn.LayerNorm(self.embed_dim))
        
        # More expressive pooling layers
        self.pool = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.BatchNorm1d(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.BatchNorm1d(self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim // 4, num_classes)
        )
    
    def forward(self, x):
        # x is spectral embeddings (batch, channels, clusters)
        batch_size, num_channels, _ = x.shape
        
        # Apply embedding layers
        h = x
        
        # First linear layer
        h = self.embed[0](h)  # (batch, channels, dim)
        
        # BatchNorm requires channel dimension to be feature dimension
        h = h.permute(0, 2, 1)  # (batch, dim, channels)
        h = self.embed[1](h)  # Apply BatchNorm
        h = h.permute(0, 2, 1)  # (batch, channels, dim)
        
        # Apply GELU activation
        h = self.embed[2](h)
        
        # Second linear layer
        h = self.embed[3](h)  # (batch, channels, embed_dim)
        
        # Apply BatchNorm again
        h = h.permute(0, 2, 1)  # (batch, embed_dim, channels)
        h = self.embed[4](h)
        h = h.permute(0, 2, 1)  # (batch, channels, embed_dim)
        
        # Reshape for attention: (channels, batch, embed_dim)
        h = h.permute(1, 0, 2)
        
        # Apply graph attention layers with FFN blocks
        for i in range(len(self.gat_layers)):
            # Multi-head attention
            attn_out, _ = self.gat_layers[i](h, h, h)
            
            # Apply batch norm to attention output
            attn_bn = attn_out.permute(1, 2, 0)  # (batch, embed_dim, channels)
            attn_bn = self.batch_norms[i](attn_bn)
            attn_bn = attn_bn.permute(2, 0, 1)  # (channels, batch, embed_dim)
            
            # Residual connection and layer norm
            h_attn = self.layer_norms[i](h + attn_bn)
            
            # Apply FFN block
            # First, reshape for the sequential FFN
            h_ffn = h_attn.permute(1, 0, 2)  # (batch, channels, embed_dim)
            
            # Handle each layer in FFN separately due to BatchNorm1d
            ffn = self.ffn_layers[i]
            
            # Linear
            h_ffn = ffn[0](h_ffn)
            
            # BatchNorm
            h_ffn = h_ffn.permute(0, 2, 1)  # (batch, embed_dim*4, channels)
            h_ffn = ffn[1](h_ffn)
            h_ffn = h_ffn.permute(0, 2, 1)  # (batch, channels, embed_dim*4)
            
            # GELU and Dropout
            h_ffn = ffn[2](h_ffn)
            h_ffn = ffn[3](h_ffn)
            
            # Linear
            h_ffn = ffn[4](h_ffn)
            
            # BatchNorm
            h_ffn = h_ffn.permute(0, 2, 1)  # (batch, embed_dim, channels) 
            h_ffn = ffn[5](h_ffn)
            h_ffn = h_ffn.permute(0, 2, 1)  # (batch, channels, embed_dim)
            
            # Convert back to (channels, batch, embed_dim) for residual
            h_ffn = h_ffn.permute(1, 0, 2)
            
            # Residual connection and layer norm
            h = self.ffn_norms[i](h_attn + h_ffn)
        
        # Global pooling across channels with weighted attention
        h_global = h.mean(0)  # (batch, embed_dim)
        
        # Apply pooling layers manually for BatchNorm
        h = self.pool[0](h_global)  # LayerNorm
        h = self.pool[1](h)  # Linear
        h = self.pool[2](h)  # BatchNorm1d
        h = self.pool[3](h)  # GELU
        h = self.pool[4](h)  # Dropout
        h = self.pool[5](h)  # Linear
        h = self.pool[6](h)  # BatchNorm1d
        h = self.pool[7](h)  # GELU
        h = self.pool[8](h)  # Dropout
        
        # Apply classifier layers
        h = self.classifier[0](h)  # Linear
        h = self.classifier[1](h)  # BatchNorm1d
        h = self.classifier[2](h)  # GELU
        h = self.classifier[3](h)  # Dropout
        out = self.classifier[4](h)  # Linear
        
        return out

# Update the create_model function to include the enhanced graph model
def create_model(model_type='transformer', num_channels=19, num_classes=2, dim=128, num_clusters=5):
    """
    Create a model of the specified type
    """
    if model_type == 'transformer':
        model = SpectralTransformerEEG(num_channels, num_classes, dim, num_clusters)
    elif model_type == 'graph':
        # Use more layers for the graph model
        model = SpectralGraphTransformer(
            num_channels=num_channels, 
            num_classes=num_classes, 
            dim=dim,  # Will be doubled inside the model
            num_clusters=num_clusters,
            num_layers=4  # Increased from 3
        )
    elif model_type == 'multiscale':
        scales = [3, 4, 5] if isinstance(num_clusters, int) else num_clusters
        # model = SpectralAlzheimersClassifier(num_channels, num_classes, dim, scales)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if hasattr(model, 'init_weights'):
        model.init_weights()
    
    return model

# Keep the original create_model function with the same interface
# def create_model(model_type='transformer', num_channels=19, num_classes=2, dim=128, num_clusters=5):
#     """
#     Create a model of the specified type
#     """
#     if model_type == 'transformer':
#         model = SpectralTransformerEEG(num_channels, num_classes, dim, num_clusters)
#     elif model_type == 'graph':
#         model = SpectralGraphTransformer(num_channels, num_classes, dim, num_clusters)
#     elif model_type == 'multiscale':
#         scales = [3, 4, 5] if isinstance(num_clusters, int) else num_clusters
#         model = SpectralAlzheimersClassifier(num_channels, num_classes, dim, scales)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")
    
#     if hasattr(model, 'init_weights'):
#         model.init_weights()
    
#     return model

# Preserve the count_parameters function
def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)