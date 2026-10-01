import torch
import numpy as np

class CrossModalAttentionVisualizer:
    """
    Computes cross-modal attention weights showing which modality drove which prediction,
    formatted for radial / chord diagram visualization.
    """
    def __init__(self, modality_names=None):
        self.modality_names = modality_names or ['Gaze', 'Handwriting', 'Speech', 'Drawing', 'EEG']

    def compute_attention_weights(self, fused_tokens):
        """
        Calculates pair-wise cross-attention cosine similarity matrix between modality tokens.
        fused_tokens shape: (B, 5, 64)
        """
        tokens = fused_tokens[0] # (5, 64)
        norm_tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)
        sim_matrix = torch.matmul(norm_tokens, norm_tokens.T).detach().cpu().numpy()
        
        # Softmax normalize matrix rows
        exp_mat = np.exp(sim_matrix * 3.0)
        attn_weights = exp_mat / np.sum(exp_mat, axis=-1, keepdims=True)
        
        # Calculate modality importance contribution weights
        importance_vec = np.mean(attn_weights, axis=0)
        importance_vec = importance_vec / np.sum(importance_vec)
        
        contributions = {self.modality_names[i]: round(float(importance_vec[i]), 4) for i in range(len(self.modality_names))}
        
        return {
            'attention_matrix': np.round(attn_weights, 4).tolist(),
            'modality_contributions': contributions
        }
