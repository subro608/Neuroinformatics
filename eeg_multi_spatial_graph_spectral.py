import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MVTEncoder(nn.Module):
    def __init__(self, dim, num_heads=4):
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
    def __init__(self, dim, num_heads=4):
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

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, num_channels, dim):
        super().__init__()
        # Spatial convolution - captures relationships between adjacent electrodes
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, dim//2, kernel_size=(num_channels, 1), stride=1),
            nn.BatchNorm2d(dim//2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Channel-wise processing
        self.channel_conv = nn.Sequential(
            nn.Conv1d(num_channels, dim//2, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim//2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Transformer encoder for spatial patterns
        self.spatial_encoder = MVTEncoder(dim)
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size, num_channels, time_length = x.shape
        
        # Spatial convolution across channels
        x_spatial = x.unsqueeze(1)  # (batch, 1, channels, time)
        spatial_features = self.spatial_conv(x_spatial)  # (batch, dim//2, 1, time)
        spatial_features = spatial_features.squeeze(2)  # (batch, dim//2, time)
        
        # Channel-wise processing
        channel_features = self.channel_conv(x)  # (batch, dim//2, time)
        
        # Combine the features
        combined_features = torch.cat([spatial_features, channel_features], dim=1)  # (batch, dim, time)
        
        # Reshape for transformer: (time, batch, dim)
        combined_features = combined_features.permute(2, 0, 1)
        
        # Process with transformer
        spatial_encoded = self.spatial_encoder(combined_features)
        
        return spatial_encoded

class MultiScaleGraphModule(nn.Module):
    def __init__(self, num_channels, dim, scales=[3, 4, 5], num_layers=4):
        super().__init__()
        self.scales = scales
        
        # Set a fixed number of heads that will work with small dimensions
        num_heads = 1  # Use 1 head to work with any dimension
        
        # Set embed_dim to a multiple of num_heads
        self.embed_dim = dim  # Use the dimension passed in
        
        # Calculate scale dimension (at least 1)
        self.scale_dim = max(1, self.embed_dim // len(scales))
        
        # Now recalculate embed_dim to ensure consistency
        self.embed_dim = self.scale_dim * len(scales)
        
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
        num_heads = 1
        
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
        
        # Get scale attention weights
        h_global_for_attn = h.mean(1)  # (batch, embed_dim)
        scale_attention_weights = self.scale_attention(h_global_for_attn)  # (batch, num_scales)
        
        # Apply attention weights to each scale output
        weighted_outputs = []
        for i, scale_output in enumerate(scale_outputs):
            weight = scale_attention_weights[:, i:i+1].unsqueeze(1)  # (batch, 1, 1)
            weighted_output = scale_output * weight
            weighted_outputs.append(weighted_output)
        
        # Recombine with attention weights
        h = torch.cat(weighted_outputs, dim=2)  # (batch, channels, embed_dim)
        
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

class MVTSpatialSpectralModel(nn.Module):
    def __init__(self, num_channels=19, num_classes=3, dim=128, scales=[3, 4, 5], num_graph_layers=4):
        super().__init__()
        
        # Spatial domain branch
        self.spatial_extractor = SpatialFeatureExtractor(num_channels, dim)
        
        # Multi-scale graph branch
        self.graph_module = MultiScaleGraphModule(
            num_channels=num_channels, 
            dim=dim, 
            scales=scales, 
            num_layers=num_graph_layers
        )
        
        # Cross-attention modules for feature interaction
        self.spatial_to_graph_attention = CrossViewAttention(dim, num_heads=4)
        self.graph_to_spatial_attention = CrossViewAttention(dim, num_heads=4)
        
        # Add second layer of cross-attention for deeper interaction
        self.spatial_to_graph_attention2 = CrossViewAttention(dim, num_heads=4)
        self.graph_to_spatial_attention2 = CrossViewAttention(dim, num_heads=4)
        
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
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.apply(_init_weights)
    
    def forward(self, x, embeddings_dict):
        batch_size = x.shape[0]
        
        # Process the first chunk for memory efficiency
        chunk_size = min(1000, x.shape[2])
        x_chunk = x[:, :, :chunk_size]
        
        # Process spatial domain
        spatial_features = self.spatial_extractor(x_chunk)  # (seq, batch, dim)
        spatial_seq_len = spatial_features.shape[0]
        
        # Process graph domain
        graph_features = self.graph_module(embeddings_dict)  # (batch, dim)
        
        # Ensure graph features have the same dimension as spatial features
        graph_feature_dim = graph_features.size(1)
        if graph_feature_dim != spatial_features.size(2):
            # Add padding or use projection to match dimensions
            graph_features = nn.functional.pad(
                graph_features, 
                (0, spatial_features.size(2) - graph_feature_dim)
            )
        
        # Create pseudo-sequence for graph features to match spatial features sequence length
        graph_features_expanded = graph_features.unsqueeze(0).expand(spatial_seq_len, -1, -1)  # (seq, batch, dim)
        
        # Apply simplified cross-attention
        spatial_attended = self.spatial_to_graph_attention(
            spatial_features, graph_features_expanded, graph_features_expanded
        )
        
        graph_attended = self.graph_to_spatial_attention(
            graph_features_expanded, spatial_features, spatial_features
        )
        
        # Pool spatial features
        spatial_pooled = spatial_attended.mean(0)  # (batch, dim)
        graph_pooled = graph_attended.mean(0)  # (batch, dim)
        
        # Print dimension info for debugging
        # print(f"Spatial pooled: {spatial_pooled.shape}, Graph pooled: {graph_pooled.shape}")
        
        # Ensure both features have same dimensionality before concatenation
        if spatial_pooled.size(1) != graph_pooled.size(1):
            # Use the smaller dimension and truncate the larger one
            min_dim = min(spatial_pooled.size(1), graph_pooled.size(1))
            spatial_pooled = spatial_pooled[:, :min_dim]
            graph_pooled = graph_pooled[:, :min_dim]
            
            # Concatenate the pooled features
            combined = torch.cat([spatial_pooled, graph_pooled], dim=1)  # (batch, min_dim*2)
            
            # Also update the fusion layer's input dimension if this is the first run
            if self.fusion[0].normalized_shape[0] != min_dim * 2:
                self.fusion[0] = nn.LayerNorm(min_dim * 2)
                self.fusion[1] = nn.Linear(min_dim * 2, min_dim)
        else:
            # Original concatenation if dimensions match
            combined = torch.cat([spatial_pooled, graph_pooled], dim=1)  # (batch, dim*2)
        
        # Apply fusion
        fused = self.fusion(combined)  # (batch, dim)
        
        # Classification
        output = self.classifier(fused)
        
        return output

# Function to create the model
def create_model(num_channels=19, num_classes=3, dim=256, scales=[3, 4, 5]):
    model = MVTSpatialSpectralModel(
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

if __name__ == "__main__":
    # Testing model creation
    model = create_model()
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test with random inputs
    if torch.cuda.is_available(): # test batch size based on GPU availability
        batch_size = 8
    else:
        batch_size = 2
    time_length = 1000
    num_channels = 19
    x = torch.randn(batch_size, num_channels, time_length)
    embeddings_dict = {
        'scale_3': torch.randn(batch_size, num_channels, 3),
        'scale_4': torch.randn(batch_size, num_channels, 4),
        'scale_5': torch.randn(batch_size, num_channels, 5)
    }
    
    # Move to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = x.to(device)
    embeddings_dict = {k: v.to(device) for k, v in embeddings_dict.items()}
    
    print(f"Using device: {device}")
    with torch.no_grad():
        outputs = model(x, embeddings_dict)
    
    print(f"Output shape: {outputs.shape}")