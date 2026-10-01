import torch
import torch.nn as nn

class SpeechEncoder(nn.Module):
    """
    Encoder for Speech & Read-Aloud Audio Modality.
    Processes prosody metrics (pause ratio, pitch variance, phoneme error rate, WPM)
    via deep MLP blocks to output unified speech embeddings.
    """
    def __init__(self, speech_dim=8, feature_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(speech_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        return self.mlp(feats)

class StandaloneSpeechClassifier(nn.Module):
    """Standalone classifier for baseline evaluation of Speech modality alone."""
    def __init__(self, speech_dim=8, feature_dim=64, num_tasks=4):
        super().__init__()
        self.encoder = SpeechEncoder(speech_dim, feature_dim)
        self.predictor = nn.Linear(feature_dim, num_tasks)
        
    def forward(self, feats):
        emb = self.encoder(feats)
        return torch.sigmoid(self.predictor(emb)), emb
