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

class CrossViewAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
        # Add BatchNorm for improved training
        self.bn = nn.BatchNorm1d(dim)
        
    def forward(self, q, k, v):
        # q, k, v shapes: (seq_len, batch, dim)
        q = self.pre_norm(q)
        att_out, _ = self.attention(q, k, v)
        
        # Apply BatchNorm1d
        batch_size = q.shape[1]
        att_out_bn = att_out.permute(1, 2, 0)  # (batch, dim, seq_len)
        att_out_bn = self.bn(att_out_bn).permute(2, 0, 1)  # Back to (seq_len, batch, dim)
        
        return self.norm(q + self.dropout(att_out_bn))

class TimeFeatureExtractor(nn.Module):
    def __init__(self, num_channels, dim):
        super().__init__()
        self.time_conv = nn.Sequential(
            nn.Conv1d(num_channels, dim, kernel_size=64, stride=32),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.time_encoder = MVTEncoder(dim)
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        time_features = self.time_conv(x)
        time_features = time_features.permute(2, 0, 1)  # (seq, batch, dim)
        time_features = self.time_encoder(time_features)
        return time_features

class MultiScaleGraphModule(nn.Module):
    def __init__(self, num_channels, dim, scales=[3, 4, 5], num_layers=4):
        super().__init__()
        self.scales = scales
        self.embed_dim = dim * 2  # Double the dimension for more capacity
        
        # Calculate the embedding dimension for each scale path
        self.scale_dim = self.embed_dim // len(scales)
        
        # Create separate embedding paths for each scale
        self.scale_embeddings = nn.ModuleList()
        for scale in scales:
            scale_embed = nn.Sequential(
                nn.Linear(scale, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Linear(dim, self.scale_dim),
                nn.BatchNorm1d(self.scale_dim)
            )
            self.scale_embeddings.append(scale_embed)
        
        # Scale attention weights to focus on the most informative scale
        self.scale_attention = nn.Sequential(
            nn.Linear(self.embed_dim, len(scales)),
            nn.Softmax(dim=-1)
        )
        
        # Increase number of heads for more expressive attention
        num_heads = 8
        
        # Deeper graph attention structure
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # Multi-head attention
            self.gat_layers.append(nn.MultiheadAttention(self.embed_dim, num_heads=num_heads, dropout=0.1))
            self.layer_norms.append(nn.LayerNorm(self.embed_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.embed_dim))
            
            # Add FFN after each attention layer
            ffn = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 4),
                nn.BatchNorm1d(self.embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.embed_dim * 4, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim)
            )
            self.ffn_layers.append(ffn)
            self.ffn_norms.append(nn.LayerNorm(self.embed_dim))
        
        # Global pooling
        self.pool = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.BatchNorm1d(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, embeddings_dict):
        # Handle multi-scale input from dictionary
        batch_size = next(iter(embeddings_dict.values())).shape[0]
        num_channels = next(iter(embeddings_dict.values())).shape[1]
        
        # Process each scale separately
        scale_outputs = []
        raw_scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            scale_key = f'scale_{scale}'
            if scale_key not in embeddings_dict:
                continue
                
            scale_data = embeddings_dict[scale_key]  # (batch, channels, scale)
            raw_scale_outputs.append(scale_data.reshape(batch_size, -1))  # Flatten for skip connection
            
            # Apply scale-specific embedding
            h = scale_data
            # First linear layer
            h = self.scale_embeddings[i][0](h)  # (batch, channels, dim)
            
            # BatchNorm
            h = h.permute(0, 2, 1)  # (batch, dim, channels)
            h = self.scale_embeddings[i][1](h)
            h = h.permute(0, 2, 1)  # (batch, channels, dim)
            
            # GELU
            h = self.scale_embeddings[i][2](h)
            
            # Second linear layer
            h = self.scale_embeddings[i][3](h)
            
            # BatchNorm
            h = h.permute(0, 2, 1)  # (batch, scale_dim, channels)
            h = self.scale_embeddings[i][4](h)
            h = h.permute(0, 2, 1)  # (batch, channels, scale_dim)
            
            scale_outputs.append(h)
        
        # Concatenate embeddings from all scales
        if scale_outputs:
            h = torch.cat(scale_outputs, dim=2)  # (batch, channels, embed_dim)
            if h.size(2) != self.embed_dim:
                # Pad to match expected dimension
                padding = torch.zeros(h.size(0), h.size(1), self.embed_dim - h.size(2), device=h.device)
                h = torch.cat([h, padding], dim=2)
        else:
            # Fallback in case no scales are available
            raise ValueError("No valid scales found in embeddings_dict")
        
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
        
        # Global pooling across channels
        h_global = h.mean(0)  # (batch, embed_dim)
        
        # Apply pooling layers manually for BatchNorm
        h = self.pool[0](h_global)  # LayerNorm
        h = self.pool[1](h)  # Linear
        h = self.pool[2](h)  # BatchNorm1d
        h = self.pool[3](h)  # GELU
        h = self.pool[4](h)  # Dropout
        
        return h  # (batch, embed_dim//2)

class MVTTimeMultiScaleModel(nn.Module):
    def __init__(self, num_channels=19, num_classes=3, dim=128, scales=[3, 4, 5], num_graph_layers=4):
        super().__init__()
        
        # Time domain branch
        self.time_extractor = TimeFeatureExtractor(num_channels, dim)
        
        # Multi-scale graph branch
        self.graph_module = MultiScaleGraphModule(
            num_channels=num_channels, 
            dim=dim, 
            scales=scales, 
            num_layers=num_graph_layers
        )
        
        # Cross-attention modules for feature interaction
        self.time_to_graph_attention = CrossViewAttention(dim)
        self.graph_to_time_attention = CrossViewAttention(dim)
        
        # Fusion module to combine both views
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim + dim),
            nn.Linear(dim + dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, num_classes)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.apply(_init_weights)
    
    def forward(self, x, embeddings_dict):
        """
        Forward pass through the network
        
        Args:
            x: Time domain signal of shape (batch, channels, time)
            embeddings_dict: Dictionary of spectral embeddings at different scales
                             {scale_3: tensor, scale_4: tensor...}
        """
        batch_size = x.shape[0]
        
        # Process time domain
        time_features = self.time_extractor(x)  # (seq, batch, dim)
        time_seq_len = time_features.shape[0]
        
        # Process graph domain
        graph_features = self.graph_module(embeddings_dict)  # (batch, dim//2)
        
        # Create pseudo-sequence for graph features to match time features sequence length
        graph_features_expanded = graph_features.unsqueeze(0).expand(time_seq_len, -1, -1)  # (seq, batch, dim//2)
        
        # Cross-attention between domains
        time_seq_len = time_features.shape[0]
        
        # Match dimensions for cross-attention by projecting graph features
        graph_features_projected = nn.functional.pad(
            graph_features_expanded, 
            (0, time_features.size(2) - graph_features_expanded.size(2))
        )  # (seq, batch, dim)
        
        # Apply cross-attention
        time_attended = self.time_to_graph_attention(time_features, graph_features_projected, graph_features_projected)
        graph_attended = self.graph_to_time_attention(graph_features_projected, time_features, time_features)
        
        # Pool time features
        time_pooled = time_attended.mean(0)  # (batch, dim)
        graph_pooled = graph_attended.mean(0)  # (batch, dim)
        
        # Concatenate the pooled features
        combined = torch.cat([time_pooled, graph_features], dim=1)  # (batch, dim + dim//2)
        
        # Apply fusion
        fused = self.fusion(combined)  # (batch, dim)
        
        # Classification
        output = self.classifier(fused)
        
        return output

# Function to create the model
def create_model(num_channels=19, num_classes=3, dim=128, scales=[3, 4, 5]):
    model = MVTTimeMultiScaleModel(
        num_channels=num_channels,
        num_classes=num_classes,
        dim=dim,
        scales=scales,
        num_graph_layers=4
    )
    return model

# Helper function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)