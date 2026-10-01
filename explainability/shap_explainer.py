import numpy as np
import torch

class TabularFeatureImportanceExplainer:
    """
    Feature importance & perturbation explainer (SHAP surrogate) for tabular & time-series features
    in NeuroLens (Gaze metrics, Speech prosody, EEG band powers).
    """
    def __init__(self, feature_names=None):
        self.feature_names = feature_names or [
            "gaze_fixation_duration", "gaze_regression_count", "gaze_saccade_velocity",
            "speech_pause_ratio", "speech_pitch_variance", "speech_phoneme_error", "speech_wpm",
            "eeg_theta_beta_ratio"
        ]

    def compute_importance_scores(self, model, sample_batch, target_head_idx=0):
        """
        Calculates feature importance vector using gradient x input perturbation analysis.
        """
        model.eval()
        # Enable gradient tracking on tabular inputs
        gaze_f = sample_batch['gaze_feats'].clone().detach().requires_grad_(True)
        speech_f = sample_batch['speech_feats'].clone().detach().requires_grad_(True)
        eeg_f = sample_batch['eeg_feats'].clone().detach().requires_grad_(True)
        
        batch = {
            'handwriting_img': sample_batch['handwriting_img'],
            'handwriting_ts': sample_batch['handwriting_ts'],
            'gaze_img': sample_batch['gaze_img'],
            'gaze_feats': gaze_f,
            'speech_feats': speech_f,
            'drawing_img': sample_batch['drawing_img'],
            'eeg_feats': eeg_f
        }
        
        outputs = model(batch)
        score = outputs['probs'][0, target_head_idx]
        score.backward()
        
        # Calculate feature attribution magnitude
        g_attr = torch.abs(gaze_f.grad[0] * gaze_f[0]).detach().cpu().numpy()
        s_attr = torch.abs(speech_f.grad[0] * speech_f[0]).detach().cpu().numpy()
        e_attr = torch.abs(eeg_f.grad[0] * eeg_f[0]).detach().cpu().numpy()
        
        # Combine selected key tabular features into SHAP-style dictionary
        importance = {
            "gaze_fixation_duration": float(g_attr[0]),
            "gaze_regression_count": float(g_attr[1]),
            "gaze_saccade_velocity": float(g_attr[2]),
            "speech_pause_ratio": float(s_attr[0]),
            "speech_pitch_variance": float(s_attr[1]),
            "speech_phoneme_error": float(s_attr[2]),
            "speech_wpm": float(s_attr[3]),
            "eeg_theta_beta_ratio": float(e_attr[4])
        }
        
        # Normalize to sum to 1.0 for clean visualization
        total = sum(importance.values()) + 1e-6
        normalized = {k: round(v / total, 4) for k, v in importance.items()}
        return normalized
