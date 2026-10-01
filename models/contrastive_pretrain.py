import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    InfoNCE Contrastive Loss for Self-Supervised Pre-Training of Biometric Encoders.
    Maximizes cosine similarity between augmented views of the same student profile (positive pair)
    while minimizing similarity against other student profiles in the batch (negative pairs).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: Normalized projections of shape (B, D) from two augmented views.
        """
        B = z_i.size(0)
        
        # Normalize representations
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate features
        representations = torch.cat([z_i, z_j], dim=0) # (2B, D)
        
        # Cosine similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature # (2B, 2B)
        
        # Mask out self-contrast
        mask = torch.eye(2 * B, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
        
        # Targets: z_i[k] matches z_j[k] which is at index k + B
        target_i = torch.arange(B, 2 * B, device=z_i.device)
        target_j = torch.arange(0, B, device=z_i.device)
        targets = torch.cat([target_i, target_j], dim=0)
        
        loss = F.cross_entropy(similarity_matrix, targets)
        return loss

class BiometricSimCLRPretrainer(nn.Module):
    """
    Self-Supervised SimCLR Pre-Training wrapper for NeuroLens encoders.
    """
    def __init__(self, encoder, input_dim=64, proj_dim=32):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, proj_dim)
        )
        self.criterion = InfoNCELoss(temperature=0.07)

    def augment_biometrics(self, x):
        """
        Applies stochastic biometric augmentation (Gaussian noise + random scaling).
        """
        noise = torch.randn_like(x) * 0.05
        scaling = torch.rand_like(x) * 0.2 + 0.9 # Scale between 0.9 and 1.1
        return (x + noise) * scaling

    def forward(self, x):
        # Generate two augmented views
        v1 = self.augment_biometrics(x)
        v2 = self.augment_biometrics(x)
        
        # Encode views
        h1 = self.encoder(v1)
        h2 = self.encoder(v2)
        
        # Project representations
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        loss = self.criterion(z1, z2)
        return loss

if __name__ == '__main__':
    dummy_encoder = nn.Linear(64, 64)
    pretrainer = BiometricSimCLRPretrainer(dummy_encoder, input_dim=64, proj_dim=32)
    
    sample_biometrics = torch.randn(16, 64)
    loss = pretrainer(sample_biometrics)
    
    print("Biometric SimCLR Pre-Training Execution Successful!")
    print(f"InfoNCE Contrastive Loss: {loss.item():.4f}")
