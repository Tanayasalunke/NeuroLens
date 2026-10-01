import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from data.loaders import get_dataloader
from data.synthetic_generator.synthetic_engine import SyntheticCognitiveGenerator

def run_eda_and_tsne_validation():
    print("=== Phase 1: Data Pipeline & Synthetic Distribution Validation ===")
    os.makedirs('notebooks', exist_ok=True)
    
    # 1. Load real sample dataset batch
    loader = get_dataloader(split='Train', batch_size=64, max_samples=300)
    real_features_list = []
    real_categories = []
    
    for batch in loader:
        # Extract multimodal features into flat vectors
        hw_ts = batch['handwriting_ts'].view(batch['handwriting_ts'].size(0), -1).numpy()
        gaze_f = batch['gaze_feats'].numpy()
        speech_f = batch['speech_feats'].numpy()
        eeg_f = batch['eeg_feats'].numpy()
        
        multimodal_vecs = np.concatenate([hw_ts, gaze_f, speech_f, eeg_f], axis=1)
        real_features_list.append(multimodal_vecs)
        real_categories.extend([f"Real_{c}" for c in batch['category']])
        
    real_features = np.vstack(real_features_list)
    print(f"Extracted {real_features.shape[0]} real multimodal samples of shape {real_features.shape[1]}")
    
    # 2. Generate synthetic minority-class samples using synthetic engine
    synth_gen = SyntheticCognitiveGenerator()
    synth_gen.train_cgan(epochs=30, batch_size=32)
    
    synth_features_list = []
    synth_categories = []
    
    # Synthesize rare co-occurring subtypes
    subtypes = [
        (0.9, 0.9, 0.8, 0.7, 'Synth_Severe_CoOccurring'),
        (0.1, 0.1, 0.1, 0.1, 'Synth_Control_Normal'),
        (0.85, 0.2, 0.1, 0.3, 'Synth_Isolated_Dyslexia'),
        (0.2, 0.9, 0.1, 0.3, 'Synth_Isolated_Dysgraphia'),
    ]
    
    for dys, dyg, dyc, adhd, label in subtypes:
        for _ in range(50):
            sample = synth_gen.generate_multimodal_sample(dys, dyg, dyc, adhd)
            hw_ts_f = sample['handwriting_ts'].flatten()
            gaze_f = sample['gaze_tabular']
            speech_f = sample['speech_tabular']
            eeg_f = sample['eeg_tabular']
            vec = np.concatenate([hw_ts_f, gaze_f, speech_f, eeg_f])
            synth_features_list.append(vec)
            synth_categories.append(label)
            
    synth_features = np.vstack(synth_features_list)
    print(f"Generated {synth_features.shape[0]} synthetic minority-class samples")
    
    # Combine real and synthetic for t-SNE mapping
    all_features = np.vstack([real_features, synth_features])
    all_labels = real_categories + synth_categories
    
    # Scale features for numerical stability
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_results = tsne.fit_transform(all_features_scaled)
    
    # 3. Plot t-SNE Distribution Overlap
    plt.figure(figsize=(12, 8), dpi=300)
    sns.set_theme(style="darkgrid")
    
    unique_labels = sorted(list(set(all_labels)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        indices = [i for i, l in enumerate(all_labels) if l == label]
        marker = 'o' if 'Real' in label else 'X'
        alpha = 0.6 if 'Real' in label else 0.85
        size = 50 if 'Real' in label else 90
        plt.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            c=[colors[idx]],
            label=label,
            alpha=alpha,
            s=size,
            marker=marker,
            edgecolors='k' if 'Synth' in label else 'none'
        )
        
    plt.title("NeuroLens Phase 1: t-SNE Multimodal Feature Overlap (Real vs Synthetic Samples)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    save_path = "notebooks/eda_synthetic_validation.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved t-SNE validation plot to {save_path}")
    
    print("=== Phase 1 Validation Complete! ===")

if __name__ == '__main__':
    run_eda_and_tsne_validation()
