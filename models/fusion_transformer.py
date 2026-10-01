import torch
import torch.nn as nn

class CrossModalFusionTransformer(nn.Module):
    """
    Cross-Modal Fusion Transformer for NeuroLens.
    Tokenizes each modality embedding into a modality token vector (dim 64),
    adds learnable modality positional embeddings, and processes tokens through
    Multi-Head Cross-Attention layers so modalities can attend to each other.
    """
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, num_modalities=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        
        # Learnable modality position embeddings
        self.modality_pos_embed = nn.Parameter(torch.randn(1, num_modalities, embed_dim) * 0.02)
        
        # Transformer Encoder Block (Self & Cross-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global fusion pooling & projection
        self.fusion_pooler = nn.Sequential(
            nn.Linear(num_modalities * embed_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, modality_tokens):
        """
        Input: modality_tokens tensor of shape (B, num_modalities, embed_dim)
        Output: fused_representation (B, embed_dim), cross_attention_weights (B, num_modalities, num_modalities)
        """
        # Add modality positional embeddings
        x = modality_tokens + self.modality_pos_embed
        
        # Pass through Transformer encoder
        fused_tokens = self.transformer(x)                         # (B, 5, 64)
        
        # Flatten tokens for global pooling
        flat = fused_tokens.reshape(fused_tokens.size(0), -1)      # (B, 320)
        fused_representation = self.fusion_pooler(flat)            # (B, 64)
        
        return fused_representation, fused_tokens
