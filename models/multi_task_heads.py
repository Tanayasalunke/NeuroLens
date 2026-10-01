import torch
import torch.nn as nn

class MultiTaskSeverityHeads(nn.Module):
    """
    Multi-task output heads for NeuroLens.
    Predicts probability and severity (0-1) for 4 cognitive condition heads:
    1. Dyslexia
    2. Dysgraphia
    3. Dyscalculia
    4. Attention-Deficit Reading Pattern
    """
    def __init__(self, embed_dim=64, num_tasks=4):
        super().__init__()
        self.num_tasks = num_tasks
        
        # Head for classification probability (0 to 1)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_tasks),
            nn.Sigmoid()
        )
        
        # Head for continuous severity level (0.0 to 1.0)
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_tasks),
            nn.Sigmoid()
        )

    def forward(self, fused_representation):
        probs = self.cls_head(fused_representation)
        severities = self.reg_head(fused_representation)
        return probs, severities

class MultiModalNeuroLensModel(nn.Module):
    """
    Complete End-to-End NeuroLens Multimodal Cognitive-Fusion Architecture.
    Combines 5 per-modality encoders, Cross-Modal Fusion Transformer, and Multi-Task Heads.
    """
    def __init__(self, feature_dim=64):
        super().__init__()
        from models.encoders.gaze_encoder import GazeEncoder
        from models.encoders.handwriting_encoder import HandwritingEncoder
        from models.encoders.speech_encoder import SpeechEncoder
        from models.encoders.drawing_encoder import DrawingEncoder
        from models.encoders.eeg_encoder import EEGEncoder
        from models.fusion_transformer import CrossModalFusionTransformer
        
        self.gaze_enc = GazeEncoder(feature_dim=feature_dim)
        self.hw_enc = HandwritingEncoder(feature_dim=feature_dim)
        self.speech_enc = SpeechEncoder(feature_dim=feature_dim)
        self.draw_enc = DrawingEncoder(feature_dim=feature_dim)
        self.eeg_enc = EEGEncoder(feature_dim=feature_dim)
        
        self.fusion = CrossModalFusionTransformer(embed_dim=feature_dim, num_modalities=5)
        self.heads = MultiTaskSeverityHeads(embed_dim=feature_dim, num_tasks=4)

    def forward(self, batch):
        gaze_tok = self.gaze_enc(batch['gaze_img'], batch['gaze_feats'])
        hw_tok = self.hw_enc(batch['handwriting_img'], batch['handwriting_ts'])
        speech_tok = self.speech_enc(batch['speech_feats'])
        draw_tok = self.draw_enc(batch['drawing_img'])
        eeg_tok = self.eeg_enc(batch['eeg_feats'])
        
        # Stack into modality tokens tensor: (B, 5, 64)
        tokens = torch.stack([gaze_tok, hw_tok, speech_tok, draw_tok, eeg_tok], dim=1)
        
        # Cross-Modal Fusion
        fused_rep, fused_tokens = self.fusion(tokens)
        
        # Multi-task predictions
        probs, severities = self.heads(fused_rep)
        
        return {
            'probs': probs,
            'severities': severities,
            'fused_representation': fused_rep,
            'fused_tokens': fused_tokens,
            'modality_tokens': tokens
        }
