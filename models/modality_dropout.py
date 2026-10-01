import torch
import torch.nn as nn

class ModalityDropout(nn.Module):
    """
    Modality Dropout (ModDrop) for Multimodal Deep Learning.
    Randomly drops (masks) entire modality tokens during training with probability `drop_prob`,
    or applies a specific boolean mask tensor during inference to simulate missing sensors
    (e.g., missing EEG or Eye Tracker).
    """
    def __init__(self, num_modalities=5, embed_dim=64, drop_prob=0.2):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        self.drop_prob = drop_prob
        # Learnable embedding token for missing/dropped modalities
        self.missing_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(self, modality_tokens, mask=None):
        """
        modality_tokens: (B, num_modalities, embed_dim)
        mask: optional boolean tensor of shape (B, num_modalities), True = Available, False = Missing
        """
        B, M, D = modality_tokens.shape
        
        if mask is not None:
            # Inference mode with explicit sensor availability mask
            # Expand mask to (B, M, D)
            mask_expanded = mask.unsqueeze(-1).expand(B, M, D)
            missing_tokens = self.missing_token.expand(B, M, D)
            output = torch.where(mask_expanded, modality_tokens, missing_tokens)
            return output, mask
        
        if self.training and self.drop_prob > 0.0:
            # Random Modality Dropout during training
            # Ensure at least 1 modality remains active per sample
            rand_mask = torch.rand(B, M, device=modality_tokens.device) > self.drop_prob
            # If all dropped for a sample, un-drop at least one random modality
            all_dropped = (~rand_mask).all(dim=1)
            if all_dropped.any():
                for i in torch.where(all_dropped)[0]:
                    rand_idx = torch.randint(0, M, (1,)).item()
                    rand_mask[i, rand_idx] = True
            
            mask_expanded = rand_mask.unsqueeze(-1).expand(B, M, D)
            missing_tokens = self.missing_token.expand(B, M, D)
            output = torch.where(mask_expanded, modality_tokens, missing_tokens)
            return output, rand_mask
        
        return modality_tokens, torch.ones(B, M, dtype=torch.bool, device=modality_tokens.device)

if __name__ == '__main__':
    mod_drop = ModalityDropout(num_modalities=5, embed_dim=64, drop_prob=0.3)
    x = torch.randn(4, 5, 64)
    
    # Test training mode
    mod_drop.train()
    out_tr, mask_tr = mod_drop(x)
    print("Training ModDrop output shape:", out_tr.shape, "Active mask sums:", mask_tr.sum(dim=1).tolist())
    
    # Test inference mode with 2 missing sensors (e.g., Gaze and EEG missing)
    mod_drop.eval()
    explicit_mask = torch.tensor([[True, True, True, False, False],
                                  [True, False, True, True, True],
                                  [False, True, True, True, False],
                                  [True, True, True, True, True]])
    out_eval, mask_eval = mod_drop(x, mask=explicit_mask)
    print("Eval ModDrop output shape:", out_eval.shape, "Mask passed successfully!")
