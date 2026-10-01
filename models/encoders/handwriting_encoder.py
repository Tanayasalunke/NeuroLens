import torch
import torch.nn as nn

class HandwritingEncoder(nn.Module):
    """
    Encoder for Handwriting & Stylus dynamics.
    Fuses 2D CNN on handwriting stroke images (e.g. Gambo letter images)
    with 1D CNN + BiLSTM on dynamic stylus time-series (pressure, velocity, tremor).
    """
    def __init__(self, ts_channels=4, feature_dim=64):
        super().__init__()
        # 2D CNN for 64x64 grayscale letter images
        self.img_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # -> (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),                          # -> (64, 2, 2)
            nn.Flatten(),                                          # -> 256
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )
        
        # 1D CNN + BiLSTM for time-series stylus features (seq_len, channels)
        self.ts_conv = nn.Sequential(
            nn.Conv1d(ts_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.bilstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, 
                              batch_first=True, bidirectional=True)
        self.ts_fc = nn.Linear(64, 32)
        
        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(64 + 32, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, img, ts):
        # Image path
        img_out = self.img_cnn(img)
        
        # Time-series path: ts shape (B, seq_len, channels) -> transpose for Conv1d
        ts_transposed = ts.transpose(1, 2)                         # (B, channels, seq_len)
        conv_out = self.ts_conv(ts_transposed).transpose(1, 2)      # (B, seq_len, 32)
        lstm_out, (hn, _) = self.bilstm(conv_out)                   # lstm_out: (B, seq_len, 64)
        ts_last = lstm_out[:, -1, :]                               # (B, 64)
        ts_out = self.ts_fc(ts_last)                               # (B, 32)
        
        concat = torch.cat([img_out, ts_out], dim=1)
        return self.head(concat)

class StandaloneHandwritingClassifier(nn.Module):
    """Standalone classifier for baseline evaluation of Handwriting modality alone."""
    def __init__(self, feature_dim=64, num_tasks=4):
        super().__init__()
        self.encoder = HandwritingEncoder(feature_dim=feature_dim)
        self.predictor = nn.Linear(feature_dim, num_tasks)
        
    def forward(self, img, ts):
        emb = self.encoder(img, ts)
        return torch.sigmoid(self.predictor(emb)), emb
