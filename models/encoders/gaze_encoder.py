import torch
import torch.nn as nn

class GazeEncoder(nn.Module):
    """
    Encoder for Eye-Tracking Gaze modality.
    Combines 2D CNN on scanpath heatmap images with MLP on tabular gaze features
    (fixation duration, regression count, saccade velocity).
    """
    def __init__(self, tabular_dim=6, feature_dim=64):
        super().__init__()
        # CNN backbone for 128x128 scanpath heatmap images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # -> (16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),                          # -> (64, 4, 4)
            nn.Flatten(),                                          # -> 1024
            nn.Linear(1024, 64),
            nn.ReLU(inplace=True)
        )
        
        # MLP for tabular gaze metrics
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True)
        )
        
        # Fusion head to unified feature vector
        self.head = nn.Sequential(
            nn.Linear(64 + 32, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, img, feats):
        img_out = self.cnn(img)
        feats_out = self.mlp(feats)
        concat = torch.cat([img_out, feats_out], dim=1)
        return self.head(concat)

class StandaloneGazeClassifier(nn.Module):
    """Standalone classifier for baseline evaluation of Gaze modality alone."""
    def __init__(self, tabular_dim=6, feature_dim=64, num_tasks=4):
        super().__init__()
        self.encoder = GazeEncoder(tabular_dim, feature_dim)
        self.predictor = nn.Linear(feature_dim, num_tasks)
        
    def forward(self, img, feats):
        emb = self.encoder(img, feats)
        return torch.sigmoid(self.predictor(emb)), emb
