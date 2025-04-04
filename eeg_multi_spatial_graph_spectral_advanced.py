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
        self.bn1 = nn.BatchNorm1d(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.BatchNorm1d(2 * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        x = self.pre_norm(x)
        att_out, _ = self.attention(x, x, x)
        batch_size = x.shape[1]
        att_out_bn = att_out.permute(1, 2, 0)
        att_out_bn = self.bn1(att_out_bn).permute(2, 0, 1)
        x = self.norm1(x + self.dropout(att_out_bn))
        
        x_ffn = x.permute(1, 0, 2)
        x_ffn = self.ffn[0](x_ffn)
        x_ffn = x_ffn.permute(0, 2, 1)
        x_ffn = self.ffn[1](x_ffn).permute(0, 2, 1)
        x_ffn = self.ffn[2](x_ffn)
        x_ffn = self.ffn[3](x_ffn)
        x_ffn = self.ffn[4](x_ffn)
        x_ffn = x_ffn.permute(0, 2, 1)
        x_ffn = self.ffn[5](x_ffn).permute(0, 2, 1)
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
        self.bn = nn.BatchNorm1d(dim)
    
    def forward(self, q, k, v):
        q = self.pre_norm(q)
        att_out, _ = self.attention(q, k, v)
        batch_size = q.shape[1]
        att_out_bn = att_out.permute(1, 2, 0)
        att_out_bn = self.bn(att_out_bn).permute(2, 0, 1)
        return self.norm(q + self.dropout(att_out_bn))

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, num_channels, dim):
        super().__init__()
        self.dim = dim  # Store full dimension
        self.half_dim = dim // 2  # Store half dimension
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, self.half_dim, kernel_size=(num_channels, 3), dilation=(1, 2)),  # Dilated convolution
            nn.BatchNorm2d(self.half_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv1d(num_channels, self.half_dim, kernel_size=1),
            nn.BatchNorm1d(self.half_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.spatial_encoders = nn.ModuleList([MVTEncoder(dim, num_heads=max(1, dim // 64)) for _ in range(3)])  # 3 layers
        
        # Learnable positional encoding (not directly used in this fixed version)
        self.pos_encoding = nn.Parameter(torch.randn(1, num_channels, self.half_dim))
    
    def forward(self, x, spatial_positions):
        batch_size, num_channels, time_length = x.shape
        x_spatial = x.unsqueeze(1)
        
        # Print shapes for debugging
        print(f"Input shape: {x.shape}")
        
        # Get spatial features
        spatial_features = self.spatial_conv(x_spatial).squeeze(2)  # (batch, half_dim, time')
        print(f"Spatial features shape: {spatial_features.shape}")
        
        # Get channel features
        channel_features = self.channel_conv(x)  # (batch, half_dim, time)
        print(f"Channel features shape: {channel_features.shape}")
        
        # IMPORTANT FIX: Make sure both tensors have the same time dimension before concatenating
        # If the time dimensions don't match, resize the smaller one
        spatial_time = spatial_features.size(2)
        channel_time = channel_features.size(2)
        
        if spatial_time != channel_time:
            print(f"Time dimension mismatch: spatial={spatial_time}, channel={channel_time}")
            if spatial_time < channel_time:
                # Resize channel_features to match spatial_features
                channel_features = channel_features[:, :, :spatial_time]
            else:
                # Resize spatial_features to match channel_features
                spatial_features = spatial_features[:, :, :channel_time]
            
            print(f"After resizing: spatial={spatial_features.shape}, channel={channel_features.shape}")
        
        # Concatenate the features
        combined_features = torch.cat([spatial_features, channel_features], dim=1)  # [batch, dim, time]
        combined_features = combined_features.permute(2, 0, 1)  # [time, batch, dim]
        
        # Pass through encoders
        for encoder in self.spatial_encoders:
            combined_features = encoder(combined_features)
        
        return combined_features

class GraphConv(nn.Module):  # Simple Graph Convolution Layer
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
    
    def forward(self, x, adj):
        # x: (batch, channels, dim), adj: (batch, channels, channels)
        x = torch.bmm(adj, x)  # Graph convolution
        x = self.linear(x)
        return self.norm(x)

class MultiScaleGraphModule(nn.Module):
    def __init__(self, num_channels, dim, scales=[3, 4, 5], num_layers=4):
        super().__init__()
        self.scales = scales
        self.embed_dim = dim
        self.scale_dim = max(1, self.embed_dim // len(scales))
        self.embed_dim = self.scale_dim * len(scales)
        
        self.scale_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scale, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Linear(dim, self.scale_dim),
                nn.BatchNorm1d(self.scale_dim)
            ) for scale in scales
        ])
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(self.scale_dim, num_heads=1) for _ in scales
        ])
        
        self.gcn = GraphConv(self.embed_dim, self.embed_dim)  # Graph convolution layer
        self.gat_layers = nn.ModuleList([
            nn.MultiheadAttention(self.embed_dim, num_heads=max(1, dim // 64), dropout=0.1)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.embed_dim) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 4),
                nn.BatchNorm1d(self.embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.embed_dim * 4, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim)
            ) for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_layers)])
        
        self.pool = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.BatchNorm1d(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def compute_dynamic_adjacency(self, scale_data):
        # scale_data: (batch, channels, scale_dim)
        adj = torch.bmm(scale_data, scale_data.transpose(1, 2))  # (batch, channels, channels)
        return F.softmax(adj, dim=-1)
    
    def forward(self, embeddings_dict):
        batch_size = next(iter(embeddings_dict.values())).shape[0]
        num_channels = next(iter(embeddings_dict.values())).shape[1]
        
        scale_outputs = []
        for i, scale in enumerate(self.scales):
            scale_key = f'scale_{scale}'
            if scale_key not in embeddings_dict:
                continue
            scale_data = embeddings_dict[scale_key]  # (batch, channels, scale)
            
            h = scale_data
            h = self.scale_embeddings[i][0](h)
            h = h.permute(0, 2, 1)
            h = self.scale_embeddings[i][1](h).permute(0, 2, 1)
            h = self.scale_embeddings[i][2](h)
            h = self.scale_embeddings[i][3](h)
            h = h.permute(0, 2, 1)
            h = self.scale_embeddings[i][4](h).permute(0, 2, 1)
            
            h = h.permute(1, 0, 2)  # (channels, batch, scale_dim)
            attn_out, _ = self.scale_attentions[i](h, h, h)
            h = attn_out.permute(1, 0, 2)  # (batch, channels, scale_dim)
            scale_outputs.append(h)
        
        h = torch.cat(scale_outputs, dim=2)  # (batch, channels, embed_dim)
        if h.size(2) != self.embed_dim:
            padding = torch.zeros(h.size(0), h.size(1), self.embed_dim - h.size(2), device=h.device)
            h = torch.cat([h, padding], dim=2)
        
        adj = self.compute_dynamic_adjacency(h)  # Dynamic adjacency
        h = self.gcn(h, adj)  # Graph convolution
        
        h = h.permute(1, 0, 2)  # (channels, batch, embed_dim)
        for i in range(len(self.gat_layers)):
            attn_out, _ = self.gat_layers[i](h, h, h)
            attn_bn = attn_out.permute(1, 2, 0)
            attn_bn = self.batch_norms[i](attn_bn).permute(2, 0, 1)
            h_attn = self.layer_norms[i](h + attn_bn)
            
            h_ffn = h_attn.permute(1, 0, 2)
            h_ffn = self.ffn_layers[i][0](h_ffn)
            h_ffn = h_ffn.permute(0, 2, 1)
            h_ffn = self.ffn_layers[i][1](h_ffn).permute(0, 2, 1)
            h_ffn = self.ffn_layers[i][2](h_ffn)
            h_ffn = self.ffn_layers[i][3](h_ffn)
            h_ffn = self.ffn_layers[i][4](h_ffn)
            h_ffn = h_ffn.permute(0, 2, 1)
            h_ffn = self.ffn_layers[i][5](h_ffn).permute(0, 2, 1)
            h_ffn = h_ffn.permute(1, 0, 2)
            h = self.ffn_norms[i](h_attn + h_ffn)
        
        h_global = h.mean(0)  # (batch, embed_dim)
        h = self.pool[0](h_global)
        h = self.pool[1](h)
        h = self.pool[2](h)
        h = self.pool[3](h)
        h = self.pool[4](h)
        return h  # (batch, embed_dim//2)

class MVTSpatialSpectralModel(nn.Module):
    def __init__(self, num_channels=19, num_classes=3, dim=128, scales=[3, 4, 5], num_graph_layers=4):
        super().__init__()
        self.spatial_extractor = SpatialFeatureExtractor(num_channels, dim)
        self.graph_module = MultiScaleGraphModule(num_channels, dim, scales, num_graph_layers)
        
        self.spatial_to_graph_attention = CrossViewAttention(dim, num_heads=max(1, dim // 32))
        self.graph_to_spatial_attention = CrossViewAttention(dim, num_heads=max(1, dim // 32))
        self.spatial_to_graph_attention2 = CrossViewAttention(dim, num_heads=max(1, dim // 32))
        self.graph_to_spatial_attention2 = CrossViewAttention(dim, num_heads=max(1, dim // 32))
        
        self.pool_attention = nn.MultiheadAttention(dim, num_heads=4)  # Attention pooling
        self.fusion_attention = nn.MultiheadAttention(dim, num_heads=4)  # Attention-based fusion
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, num_classes)
        )
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
        chunk_size = min(1000, x.shape[2])
        x_chunk = x[:, :, :chunk_size]
        
        spatial_positions = embeddings_dict.get('spatial_positions', torch.zeros(batch_size, 19, 3).to(x.device))
        spatial_features = self.spatial_extractor(x_chunk, spatial_positions)  # (seq, batch, dim)
        graph_features = self.graph_module(embeddings_dict)  # (batch, dim//2)
        
        if graph_features.size(1) != spatial_features.size(2):
            graph_features = F.pad(graph_features, (0, spatial_features.size(2) - graph_features.size(1)))
        
        graph_features_expanded = graph_features.unsqueeze(0).expand(spatial_features.shape[0], -1, -1)
        
        spatial_attended = self.spatial_to_graph_attention(spatial_features, graph_features_expanded, graph_features_expanded)
        graph_attended = self.graph_to_spatial_attention(graph_features_expanded, spatial_features, spatial_features)
        
        spatial_attended2 = self.spatial_to_graph_attention2(spatial_attended, graph_attended, graph_attended)
        graph_attended2 = self.graph_to_spatial_attention2(graph_attended, spatial_attended, spatial_attended)
        
        spatial_pooled, _ = self.pool_attention(spatial_attended2[-1:], spatial_attended2, spatial_attended2)  # (1, batch, dim)
        graph_pooled, _ = self.pool_attention(graph_attended2[-1:], graph_attended2, graph_attended2)  # (1, batch, dim)
        
        spatial_pooled = spatial_pooled.squeeze(0)  # (batch, dim)
        graph_pooled = graph_pooled.squeeze(0)  # (batch, dim)
        
        combined = torch.stack([spatial_pooled, graph_pooled], dim=0)  # (2, batch, dim)
        fused, _ = self.fusion_attention(combined, combined, combined)  # (2, batch, dim)
        fused = fused.mean(0) + spatial_pooled + graph_pooled  # (batch, dim) with residual
        
        output = self.classifier(fused)
        return output

def create_model(num_channels=19, num_classes=3, dim=256, scales=[3, 4, 5]):
    model = MVTSpatialSpectralModel(num_channels, num_classes, dim, scales, num_graph_layers=4)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = create_model()
    print(f"Model created with {count_parameters(model):,} parameters")
    
    if torch.cuda.is_available():
        batch_size = 8
    else:
        batch_size = 2
    time_length = 1000
    num_channels = 19
    x = torch.randn(batch_size, num_channels, time_length)
    embeddings_dict = {
        'scale_3': torch.randn(batch_size, num_channels + 5, 3),  # Updated for lobe features
        'scale_4': torch.randn(batch_size, num_channels + 5, 4),
        'scale_5': torch.randn(batch_size, num_channels + 5, 5),
        'spatial_positions': torch.randn(batch_size, num_channels, 3)
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = x.to(device)
    embeddings_dict = {k: v.to(device) for k, v in embeddings_dict.items()}
    
    print(f"Using device: {device}")
    with torch.no_grad():
        outputs = model(x, embeddings_dict)
    print(f"Output shape: {outputs.shape}")