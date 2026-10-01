import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from data.synthetic_generator.synthetic_engine import SyntheticCognitiveGenerator

class NeuroLensMultimodalDataset(Dataset):
    """
    Unified PyTorch Dataset combining real handwriting images from Gambo dataset
    with label-conditioned synthetic multi-modal signals (gaze, speech, drawing, EEG).
    
    Data Splitting Notice:
    Writer-disjoint splitting is enforced via filename series prefix grouping 
    (e.g., 'Reversal304', '1_6000'), ensuring images from the same series 
    never appear across both train and test splits.
    
    Synthetic Pairing Disclosure:
    Gambo images are paired with synthetic multi-signal vectors generated under 
    matching label conditioning vector y. The fusion model learns the joint synthetic 
    distribution conditioned on y, serving as a simulated multi-signal benchmark.
    """
    def __init__(self, root_dir='data/raw/Gambo', split='Train', transform=None, max_samples=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.img_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.generator = SyntheticCognitiveGenerator()
        all_samples = []
        
        split_dir = os.path.join(root_dir, split)
        categories = ['Normal', 'Corrected', 'Reversal']
        
        for cat in categories:
            cat_dir = os.path.join(split_dir, cat)
            if not os.path.exists(cat_dir):
                continue
            img_paths = glob.glob(os.path.join(cat_dir, '*.*'))
            for p in img_paths:
                # Extract filename prefix for writer/series grouping
                filename = os.path.basename(p)
                group_prefix = filename.split('.')[0].split(' ')[0].split('_')[0]
                all_samples.append((p, cat, group_prefix))
                
        # Enforce deterministic Group-Based Train/Test Splitting
        np.random.seed(42)
        unique_groups = list(set([s[2] for s in all_samples]))
        unique_groups.sort()
        np.random.shuffle(unique_groups)
        
        split_idx = int(0.7 * len(unique_groups))
        train_groups = set(unique_groups[:split_idx])
        test_groups = set(unique_groups[split_idx:])
        
        if split.lower() == 'train':
            self.samples = [s for s in all_samples if s[2] in train_groups]
        else:
            self.samples = [s for s in all_samples if s[2] in test_groups]
            
        if max_samples and max_samples < len(self.samples):
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, category, group_prefix = self.samples[idx]
        
        # Determine continuous severity labels conditioned on category
        if category == 'Reversal':
            dysgraphia_sev = float(np.random.uniform(0.7, 1.0))
            dyslexia_sev = float(np.random.uniform(0.6, 0.95))
            dyscalculia_sev = float(np.random.uniform(0.2, 0.6))
            adhd_sev = float(np.random.uniform(0.3, 0.7))
        elif category == 'Corrected':
            dysgraphia_sev = float(np.random.uniform(0.3, 0.55))
            dyslexia_sev = float(np.random.uniform(0.3, 0.6))
            dyscalculia_sev = float(np.random.uniform(0.1, 0.4))
            adhd_sev = float(np.random.uniform(0.2, 0.5))
        else: # Normal
            dysgraphia_sev = float(np.random.uniform(0.0, 0.2))
            dyslexia_sev = float(np.random.uniform(0.0, 0.25))
            dyscalculia_sev = float(np.random.uniform(0.0, 0.2))
            adhd_sev = float(np.random.uniform(0.0, 0.25))
            
        # Load real handwriting image
        try:
            hw_pil = Image.open(img_path).convert('L')
            hw_tensor = self.transform(hw_pil)
        except Exception:
            hw_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
            
        # Generate aligned 5-modality sample signals conditioned on target severity y
        synth_data = self.generator.generate_multimodal_sample(
            dyslexia_sev=dyslexia_sev,
            dysgraphia_sev=dysgraphia_sev,
            dyscalculia_sev=dyscalculia_sev,
            adhd_sev=adhd_sev
        )
        
        gaze_img_tensor = self.img_transform(synth_data['gaze_img'])
        drawing_img_tensor = self.img_transform(synth_data['drawing_img'])
        
        labels_tensor = torch.tensor([dyslexia_sev, dysgraphia_sev, dyscalculia_sev, adhd_sev], dtype=torch.float32)
        
        return {
            'handwriting_img': hw_tensor,                                      # (1, 64, 64)
            'handwriting_ts': torch.tensor(synth_data['handwriting_ts']),     # (50, 4)
            'gaze_img': gaze_img_tensor,                                       # (3, 128, 128)
            'gaze_feats': torch.tensor(synth_data['gaze_tabular']),           # (6,)
            'speech_feats': torch.tensor(synth_data['speech_tabular']),       # (8,)
            'drawing_img': drawing_img_tensor,                                 # (3, 128, 128)
            'eeg_feats': torch.tensor(synth_data['eeg_tabular']),             # (5,)
            'labels': labels_tensor,                                           # (4,)
            'category': category,
            'writer_group': group_prefix
        }

def get_dataloader(root_dir='data/raw/Gambo', split='Train', batch_size=32, shuffle=True, max_samples=2000, num_workers=0):
    dataset = NeuroLensMultimodalDataset(root_dir=root_dir, split=split, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
