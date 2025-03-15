import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

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

class CrossModalAttention(nn.Module):
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

class ModalityProcessor(nn.Module):
    """Processes individual modality data"""
    def __init__(self, input_dim, output_dim, num_heads=8):
        super().__init__()
        self.embed = nn.Linear(input_dim, output_dim)
        self.encoder = MVTEncoder(output_dim, num_heads)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.norm(x)

class MultiModalMVT(nn.Module):
    def __init__(
        self,
        num_classes=3,
        dim=128,
        eegpt_model=None,
        modality_dims={
            'eeg_time': 19,      # Number of EEG channels
            'eeg_freq': 19,      # Frequency features
            'eeg_conn': 361,     # Connectivity matrix (19x19)
            'mod1': 64,          # First additional modality
            'mod2': 32,          # Second additional modality
            'mod3': 128          # Third additional modality
        },
        num_heads=8,
        dropout_rate=0.1,
        use_eegpt=True
    ):
        super().__init__()
        
        self.use_eegpt = use_eegpt
        # Store EEGPT model
        self.eegpt = eegpt_model
        if self.eegpt is not None and use_eegpt:
            for param in self.eegpt.parameters():
                param.requires_grad = False
                
        # Add EEGPT dimension if using it
        if use_eegpt and eegpt_model is not None:
            modality_dims['eegpt'] = 512  # EEGPT output dimension
        
        # Modality-specific processors
        self.modality_processors = nn.ModuleDict({
            mod_name: ModalityProcessor(dim_size, dim, num_heads)
            for mod_name, dim_size in modality_dims.items()
        })
        
        # Cross-modal attention
        num_modalities = len(modality_dims)
        self.cross_modal_attention = nn.ModuleList([
            CrossModalAttention(dim, num_heads)
            for _ in range(num_modalities)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * num_modalities),
            nn.Linear(dim * num_modalities, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim // 2, num_classes)
        )

    def process_eeg(self, eeg_dict):
        """Process EEG data using both EEGPT and traditional features"""
        processed_features = {
            'time': self.modality_processors['eeg_time'](eeg_dict['time']),
            'freq': self.modality_processors['eeg_freq'](eeg_dict['freq']),
            'conn': self.modality_processors['eeg_conn'](eeg_dict['connectivity'].flatten(1))
        }
        
        if self.use_eegpt and self.eegpt is not None:
            with torch.no_grad():
                eegpt_features = self.eegpt.forward_features(eeg_dict['time'])
                processed_features['eegpt'] = self.modality_processors['eegpt'](eegpt_features)
            
        return processed_features

    def forward(self, x_dict):
        """
        x_dict: dictionary containing:
            - 'eeg': Dictionary of EEG features
            - 'mod1': First additional modality
            - 'mod2': Second additional modality
            - 'mod3': Third additional modality
        """
        batch_size = x_dict['eeg']['time'].shape[0]
        
        # Process EEG data
        eeg_features = self.process_eeg(x_dict['eeg'])
        
        # Process other modalities
        modality_features = {
            'mod1': self.modality_processors['mod1'](x_dict['mod1']),
            'mod2': self.modality_processors['mod2'](x_dict['mod2']),
            'mod3': self.modality_processors['mod3'](x_dict['mod3'])
        }
        
        # Combine all features
        all_features = {**eeg_features, **modality_features}
        
        # Cross-modal attention
        attended_features = []
        for i, (mod_name, features) in enumerate(all_features.items()):
            other_features = [f for n, f in all_features.items() if n != mod_name]
            
            # Apply attention with other modalities
            for other_feat in other_features:
                att_out = self.cross_modal_attention[i](
                    features, other_feat, other_feat
                )
                features = features + att_out
                
            attended_features.append(features)
        
        # Concatenate and fuse features
        combined = torch.cat(attended_features, dim=-1)
        fused = self.fusion(combined)
        
        # Classification
        output = self.classifier(fused)
        
        return output

    def get_attention_weights(self, x_dict):
        """Get attention weights for visualization"""
        attention_weights = {}
        
        # Process data first
        eeg_features = self.process_eeg(x_dict['eeg'])
        modality_features = {
            'mod1': self.modality_processors['mod1'](x_dict['mod1']),
            'mod2': self.modality_processors['mod2'](x_dict['mod2']),
            'mod3': self.modality_processors['mod3'](x_dict['mod3'])
        }
        all_features = {**eeg_features, **modality_features}
        
        # Get attention weights for each modality pair
        for i, (mod1_name, mod1_features) in enumerate(all_features.items()):
            attention_weights[mod1_name] = {}
            for mod2_name, mod2_features in all_features.items():
                if mod1_name != mod2_name:
                    _, weights = self.cross_modal_attention[i].attention(
                        mod1_features, mod2_features, mod2_features
                    )
                    attention_weights[mod1_name][mod2_name] = weights
        
        return attention_weights

def create_multimodal_mvt(num_classes=3, dim=128, eegpt_model=None, modality_dims=None, num_heads=8, dropout_rate=0.1):
    """Factory function to create MultiModalMVT model"""
    if modality_dims is None:
        modality_dims = {
            'eeg_time': 19,      # Number of EEG channels
            'eeg_freq': 19,      # Frequency features
            'eeg_conn': 361,     # Connectivity matrix (19x19)
            'mod1': 64,          # First additional modality
            'mod2': 32,          # Second additional modality
            'mod3': 128          # Third additional modality
        }
    
    model = MultiModalMVT(
        num_classes=num_classes,
        dim=dim,
        eegpt_model=eegpt_model,
        modality_dims=modality_dims,
        num_heads=num_heads,
        dropout_rate=dropout_rate
    )
    
    return model

# Training helper functions
def train_step(model, optimizer, criterion, batch_data, batch_labels):
    optimizer.zero_grad()
    outputs = model(batch_data)
    loss = criterion(outputs, batch_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def validate_step(model, criterion, batch_data, batch_labels):
    with torch.no_grad():
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == batch_labels).sum().item()
    return loss.item(), correct

# Example usage:
if __name__ == "__main__":
    # Create sample data dimensions
    modality_dims = {
        'eeg_time': 19,
        'eeg_freq': 19,
        'eeg_conn': 361,
        'mod1': 64,
        'mod2': 32,
        'mod3': 128
    }
    
    # Initialize model
    model = create_multimodal_mvt(
        num_classes=3,
        dim=128,
        modality_dims=modality_dims
    )
    
    # Print model summary
    print(f"Created MultiModal MVT model with {sum(p.numel() for p in model.parameters())} parameters")