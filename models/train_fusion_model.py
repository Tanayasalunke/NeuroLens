import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from data.loaders import get_dataloader
from models.multi_task_heads import MultiModalNeuroLensModel

class NeuroLensCompositeLoss(nn.Module):
    """
    Composite Multi-Task Loss:
    Loss = alpha * BCE(classification) + beta * MSE(severity_regression) + gamma * ContrastiveLoss(embeddings)
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def contrastive_alignment_loss(self, modality_tokens):
        # modality_tokens: (B, 5, 64)
        # Encourages modality embeddings of the same child sample to align in shared latent space
        norm_tokens = nn.functional.normalize(modality_tokens, p=2, dim=-1)
        sim_matrix = torch.matmul(norm_tokens, norm_tokens.transpose(1, 2)) # (B, 5, 5)
        eye = torch.eye(5, device=modality_tokens.device).unsqueeze(0)
        off_diag_sim = sim_matrix * (1.0 - eye)
        # Maximize average off-diagonal inter-modal similarity
        return 1.0 - off_diag_sim.mean()

    def forward(self, outputs, target_severities):
        target_cls = (target_severities >= 0.5).float()
        
        loss_cls = self.bce(outputs['probs'], target_cls)
        loss_reg = self.mse(outputs['severities'], target_severities)
        loss_cont = self.contrastive_alignment_loss(outputs['modality_tokens'])
        
        total_loss = self.alpha * loss_cls + self.beta * loss_reg + self.gamma * loss_cont
        return total_loss, loss_cls, loss_reg, loss_cont

def train_fusion_model(epochs=6, batch_size=32, lr=0.001):
    print("=== Phase 3: Cross-Modal Fusion Transformer & Multi-Task Training ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    train_loader = get_dataloader(split='Train', batch_size=batch_size, max_samples=1200)
    test_loader = get_dataloader(split='Test', batch_size=batch_size, max_samples=300)
    
    model = MultiModalNeuroLensModel().to(device)
    
    # Optionally load pre-trained encoder checkpoints from Phase 2
    try:
        model.gaze_enc.load_state_dict(torch.load('checkpoints/gaze_encoder.pt', map_location=device))
        model.hw_enc.load_state_dict(torch.load('checkpoints/handwriting_encoder.pt', map_location=device))
        model.speech_enc.load_state_dict(torch.load('checkpoints/speech_encoder.pt', map_location=device))
        model.draw_enc.load_state_dict(torch.load('checkpoints/drawing_encoder.pt', map_location=device))
        model.eeg_enc.load_state_dict(torch.load('checkpoints/eeg_encoder.pt', map_location=device))
        print("Successfully loaded Phase 2 pretrained modality encoder weights into fusion model!")
    except Exception as e:
        print(f"Starting fusion training with fresh weights ({e})")
        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = NeuroLensCompositeLoss(alpha=1.0, beta=1.0, gamma=0.1)
    
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0.0
        cls_epoch_loss = 0.0
        reg_epoch_loss = 0.0
        
        for batch in train_loader:
            # Move inputs to device
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            target_severities = batch_dev['labels']
            
            optimizer.zero_grad()
            outputs = model(batch_dev)
            
            loss, l_cls, l_reg, l_cont = criterion(outputs, target_severities)
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            cls_epoch_loss += l_cls.item()
            reg_epoch_loss += l_reg.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Total Loss: {total_epoch_loss/len(train_loader):.4f} | Cls BCE: {cls_epoch_loss/len(train_loader):.4f} | Reg MSE: {reg_epoch_loss/len(train_loader):.4f}")
        
    # Evaluate on Test Set
    model.eval()
    all_preds_cls = []
    all_preds_reg = []
    all_targets_reg = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch_dev)
            
            probs = outputs['probs'].cpu().numpy()
            sevs = outputs['severities'].cpu().numpy()
            targets = batch_dev['labels'].cpu().numpy()
            
            all_preds_cls.append((probs >= 0.5).astype(np.float32))
            all_preds_reg.append(sevs)
            all_targets_reg.append(targets)
            
    all_preds_cls = np.vstack(all_preds_cls)
    all_preds_reg = np.vstack(all_preds_reg)
    all_targets_reg = np.vstack(all_targets_reg)
    all_targets_cls = (all_targets_reg >= 0.5).astype(np.float32)
    
    test_acc = accuracy_score(all_targets_cls.flatten(), all_preds_cls.flatten())
    test_f1 = f1_score(all_targets_cls.flatten(), all_preds_cls.flatten(), average='macro')
    test_rmse = np.sqrt(mean_squared_error(all_targets_reg.flatten(), all_preds_reg.flatten()))
    
    print("\n========================================================")
    print("=== NeuroLens Cross-Modal Fusion Transformer Results ===")
    print("========================================================")
    print(f"Multi-Task Classification Accuracy : {test_acc*100:.2f}%")
    print(f"Multi-Task Macro F1-Score         : {test_f1:.4f}")
    print(f"Severity Regression RMSE           : {test_rmse:.4f}")
    print("--------------------------------------------------------")
    print("Per-Condition Performance Metrics:")
    conditions = ['Dyslexia', 'Dysgraphia', 'Dyscalculia', 'Attention-Deficit']
    for i, cond in enumerate(conditions):
        c_acc = accuracy_score(all_targets_cls[:, i], all_preds_cls[:, i])
        c_f1 = f1_score(all_targets_cls[:, i], all_preds_cls[:, i], zero_division=0)
        c_rmse = np.sqrt(mean_squared_error(all_targets_reg[:, i], all_preds_reg[:, i]))
        print(f"  • {cond:<20}: Accuracy = {c_acc*100:6.2f}% | F1 = {c_f1:6.4f} | Severity RMSE = {c_rmse:.4f}")
    print("========================================================\n")
    
    # Save full model checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/neurolens_fusion_model.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved complete NeuroLens fusion model checkpoint to {checkpoint_path}")
    
    print("=== Phase 3 Execution Complete! ===")

if __name__ == '__main__':
    train_fusion_model(epochs=5)
