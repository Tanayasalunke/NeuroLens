import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import f1_score, accuracy_score

from data.loaders import get_dataloader
from models.encoders.gaze_encoder import StandaloneGazeClassifier
from models.encoders.handwriting_encoder import StandaloneHandwritingClassifier
from models.encoders.speech_encoder import StandaloneSpeechClassifier
from models.encoders.drawing_encoder import StandaloneDrawingClassifier
from models.encoders.eeg_encoder import StandaloneEEGClassifier

def evaluate_classifier(model, dataloader, modality_name, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device)
            # Binary threshold target (severity >= 0.5)
            binary_targets = (labels >= 0.5).float()
            
            if modality_name == 'gaze':
                img, feats = batch['gaze_img'].to(device), batch['gaze_feats'].to(device)
                preds, _ = model(img, feats)
            elif modality_name == 'handwriting':
                img, ts = batch['handwriting_img'].to(device), batch['handwriting_ts'].to(device)
                preds, _ = model(img, ts)
            elif modality_name == 'speech':
                feats = batch['speech_feats'].to(device)
                preds, _ = model(feats)
            elif modality_name == 'drawing':
                img = batch['drawing_img'].to(device)
                preds, _ = model(img)
            elif modality_name == 'eeg':
                feats = batch['eeg_feats'].to(device)
                preds, _ = model(feats)
                
            all_preds.append((preds >= 0.5).cpu().numpy())
            all_targets.append(binary_targets.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    acc = accuracy_score(all_targets.flatten(), all_preds.flatten())
    f1 = f1_score(all_targets.flatten(), all_preds.flatten(), average='macro', zero_division=0)
    return acc, f1

def train_and_evaluate_all_modalities(epochs=5, batch_size=32):
    print("=== Phase 2: Standalone Per-Modality Encoder Training & Benchmarking ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    train_loader = get_dataloader(split='Train', batch_size=batch_size, max_samples=800)
    test_loader = get_dataloader(split='Test', batch_size=batch_size, max_samples=200)
    
    modalities = {
        'gaze': StandaloneGazeClassifier().to(device),
        'handwriting': StandaloneHandwritingClassifier().to(device),
        'speech': StandaloneSpeechClassifier().to(device),
        'drawing': StandaloneDrawingClassifier().to(device),
        'eeg': StandaloneEEGClassifier().to(device)
    }
    
    results = {}
    criterion = nn.BCELoss()
    
    for name, model in modalities.items():
        print(f"\n--- Training Standalone {name.upper()} Encoder ---")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                labels = batch['labels'].to(device)
                target = (labels >= 0.5).float()
                
                optimizer.zero_grad()
                if name == 'gaze':
                    preds, _ = model(batch['gaze_img'].to(device), batch['gaze_feats'].to(device))
                elif name == 'handwriting':
                    preds, _ = model(batch['handwriting_img'].to(device), batch['handwriting_ts'].to(device))
                elif name == 'speech':
                    preds, _ = model(batch['speech_feats'].to(device))
                elif name == 'drawing':
                    preds, _ = model(batch['drawing_img'].to(device))
                elif name == 'eeg':
                    preds, _ = model(batch['eeg_feats'].to(device))
                    
                loss = criterion(preds, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            train_acc, train_f1 = evaluate_classifier(model, train_loader, name, device)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
            
        test_acc, test_f1 = evaluate_classifier(model, test_loader, name, device)
        results[name] = {'acc': test_acc, 'f1': test_f1}
        print(f"--> {name.upper()} Standalone Baseline: Test Accuracy = {test_acc*100:.2f}%, F1 = {test_f1:.4f}")
        
    print("\n=== Summary of Single-Modality Baseline Benchmarks ===")
    print(f"{'Modality':<15} | {'Test Accuracy':<15} | {'Test F1-Score':<15}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name.capitalize():<15} | {metrics['acc']*100:6.2f}%         | {metrics['f1']:6.4f}")
    print("=" * 50)
    
    # Save checkpoint models
    os.makedirs('checkpoints', exist_ok=True)
    for name, model in modalities.items():
        torch.save(model.encoder.state_dict(), f'checkpoints/{name}_encoder.pt')
    print("Saved single-modality encoder checkpoints to checkpoints/")
    
    return results

if __name__ == '__main__':
    train_and_evaluate_all_modalities(epochs=3)
