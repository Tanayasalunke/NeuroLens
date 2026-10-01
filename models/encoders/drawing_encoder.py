import torch
import torch.nn as nn

class DrawingEncoder(nn.Module):
    """
    Encoder for Drawing Tests (Clock-Drawing Test & Bender-Gestalt figures).
    Uses a 2D CNN backbone to extract spatial layout and visuomotor distortion features.
    """
    def __init__(self, feature_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # -> (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),                          # -> (128, 4, 4)
            nn.Flatten(),                                          # -> 2048
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        return self.cnn(img)

class StandaloneDrawingClassifier(nn.Module):
    """Standalone classifier for baseline evaluation of Drawing modality alone."""
    def __init__(self, feature_dim=64, num_tasks=4):
        super().__init__()
        self.encoder = DrawingEncoder(feature_dim)
        self.predictor = nn.Linear(feature_dim, num_tasks)
        
    def forward(self, img):
        emb = self.encoder(img)
        return torch.sigmoid(self.predictor(emb)), emb
