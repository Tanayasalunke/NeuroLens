import torch
import torch.nn as nn

class EEGEncoder(nn.Module):
    """
    Encoder for EEG spectral band-power dynamics (delta, theta, alpha, beta, theta/beta ratio).
    Processes spectral density features via 1D Conv & MLP blocks.
    """
    def __init__(self, eeg_dim=5, feature_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 * eeg_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        # feats shape (B, 5) -> reshape to (B, 1, 5) for Conv1d
        x = feats.unsqueeze(1)
        conv_out = self.conv(x).flatten(1)
        return self.mlp(conv_out)

class StandaloneEEGClassifier(nn.Module):
    """Standalone classifier for baseline evaluation of EEG modality alone."""
    def __init__(self, eeg_dim=5, feature_dim=64, num_tasks=4):
        super().__init__()
        self.encoder = EEGEncoder(eeg_dim, feature_dim)
        self.predictor = nn.Linear(feature_dim, num_tasks)
        
    def forward(self, feats):
        emb = self.encoder(feats)
        return torch.sigmoid(self.predictor(emb)), emb
