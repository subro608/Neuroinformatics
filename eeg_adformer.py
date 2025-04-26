import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.ADformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ADformerLayer
from layers.Embed import TokenChannelEmbedding
import numpy as np


class ADformerEEG(nn.Module):
    """
    ADformer (Augmented Dual-former) for EEG Classification
    Uses dual-path processing with temporal and channel dimensions
    """

    def __init__(self, configs):
        super(ADformerEEG, self).__init__()
        self.task_name = "classification"  # Fixed to classification for EEG data
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in  # Number of EEG channels (typically 19 for standard 10-20 system)
        self.seq_len = configs.seq_len  # Sequence length of EEG data
        
        # Parse configuration parameters
        if configs.no_temporal_block and configs.no_channel_block:
            raise ValueError("At least one of the two blocks should be True")
            
        # Define patch sizes for temporal dimension
        if configs.no_temporal_block:
            patch_len_list = []
        else:
            patch_len_list = list(map(int, configs.patch_len_list.split(",")))
            
        # Define upscaling dimensions for channel dimension
        if configs.no_channel_block:
            up_dim_list = []
        else:
            up_dim_list = list(map(int, configs.up_dim_list.split(",")))
            
        # Define stride settings (default: same as patch length)
        stride_list = patch_len_list
        
        # Calculate patch numbers for each patch length
        patch_num_list = [
            int((self.seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        
        # Get augmentation types
        augmentations = configs.augmentations.split(",")

        # Embedding layer - handles both temporal and channel dimensions
        self.enc_embedding = TokenChannelEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            patch_len_list,
            up_dim_list,
            stride_list,
            configs.dropout,
            augmentations,
        )
        
        # Encoder with dual-path attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ADformerLayer(
                        len(patch_len_list),
                        len(up_dim_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        
        # Classification head
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.d_model * len(patch_num_list) +
            configs.d_model * len(up_dim_list),
            configs.num_class,
        )
        
        # Additional EEG-specific components
        self.spatial_attention = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.LayerNorm(configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, 1)
        )
        
        print(f"ADformerEEG model initialized with:")
        print(f"- {len(patch_len_list)} temporal scales: {patch_len_list}")
        print(f"- {len(up_dim_list)} channel scales: {up_dim_list}")
        print(f"- Augmentations: {augmentations}")
        print(f"- Hidden dimension: {configs.d_model}")
        print(f"- Number of classes: {configs.num_class}")

    def classification(self, x_enc, x_mark_enc):
        """
        Perform classification on EEG data
        
        Args:
            x_enc: Input EEG data [batch_size, seq_len, enc_in]
            x_mark_enc: Optional marker sequence (can contain EEG metadata)
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Embedding
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        
        # Dual-path encoder processing
        enc_out_t, enc_out_c, attns_t, attns_c = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        
        # Handle cases where one path might be disabled
        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            # Combine temporal and channel representations
            enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)

        # Output projection
        output = self.act(enc_out)
        output = self.dropout(output)
        
        # Reshape for classification
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass of the ADformer model
        For EEG classification, we only need x_enc
        
        Args:
            x_enc: Input EEG data [batch_size, seq_len, enc_in]
            x_mark_enc: Optional marker sequence (can be None)
            x_dec, x_mark_dec: Not used for classification but kept for interface consistency
            mask: Optional mask (not used in normal classification)
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # For classification, only use the classification method
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        
        return None


class Config:
    """Configuration class for ADformerEEG model"""
    def __init__(self, 
                 enc_in=19,  # Number of EEG channels
                 seq_len=1000,  # EEG sequence length
                 patch_len_list="200,400,800",  # Multiple temporal scales for patching
                 up_dim_list="10,15,20",  # Multiple channel dimension scales
                 d_model=128,  # Model dimension
                 n_heads=4,  # Number of attention heads
                 e_layers=3,  # Number of encoder layers
                 d_ff=256,  # Feed-forward dimension
                 dropout=0.2,  # Dropout rate
                 activation="gelu",  # Activation function
                 output_attention=False,  # Whether to output attention maps
                 no_temporal_block=False,  # Whether to disable temporal processing
                 no_channel_block=False,  # Whether to disable channel processing
                 no_inter_attn=False,  # Whether to disable inter-attention
                 augmentations="none,jitter,scaling",  # Augmentation methods
                 num_class=3):  # Number of classes for classification
        
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.patch_len_list = patch_len_list
        self.up_dim_list = up_dim_list
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.output_attention = output_attention
        self.no_temporal_block = no_temporal_block
        self.no_channel_block = no_channel_block
        self.no_inter_attn = no_inter_attn
        self.augmentations = augmentations
        self.num_class = num_class
        self.task_name = "classification"


def create_adformer_eeg_model(
    enc_in=19,  # Number of EEG channels
    seq_len=1000,  # EEG sequence length
    patch_len_list="200,400,800",  # Multiple temporal scales for patching
    up_dim_list="10,15,20",  # Multiple channel dimension scales
    d_model=64,  # Model dimension
    n_heads=4,  # Number of attention heads
    e_layers=3,  # Number of encoder layers
    d_ff=256,  # Feed-forward dimension
    dropout=0.4,  # Dropout rate
    activation="gelu",  # Activation function
    output_attention=False,  # Whether to output attention maps
    no_temporal_block=False,  # Whether to disable temporal processing
    no_channel_block=False,  # Whether to disable channel processing
    no_inter_attn=False,  # Whether to disable inter-attention
    augmentations="none,jitter,scale,time_shift",
    num_class=3  # Number of classes for classification
):
    """Factory function to create an ADformerEEG model with specified configuration"""
    config = Config(
        enc_in=enc_in,
        seq_len=seq_len,
        patch_len_list=patch_len_list,
        up_dim_list=up_dim_list,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        output_attention=output_attention,
        no_temporal_block=no_temporal_block,
        no_channel_block=no_channel_block,
        no_inter_attn=no_inter_attn,
        augmentations=augmentations,
        num_class=num_class
    )
    
    model = ADformerEEG(config)
    return model


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage
    model = create_adformer_eeg_model(
        enc_in=19,  # 19 EEG channels (standard 10-20 system)
        seq_len=1000,  # 1000 time points
        patch_len_list="200,400,800",  # Multiple temporal scales
        up_dim_list="10,15,20",  # Multiple channel scales
        d_model=128,
        num_class=3  # 3-class classification
    )
    
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Example input
    batch_size = 8
    x_enc = torch.randn(batch_size, 1000, 19)  # [batch, seq_len, enc_in]
    x_mark_enc = torch.randn(batch_size, 1000, 1)  # Optional marker
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x_enc, x_mark_enc)
    
    print(f"Output shape: {outputs.shape}")