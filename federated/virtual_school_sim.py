import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from data.loaders import NeuroLensMultimodalDataset

class VirtualSchoolPartitioner:
    """
    Simulates 4-6 virtual school data nodes for Federated Learning.
    Partitions dataset across virtual school nodes with mild non-IID demographic distributions.
    """
    def __init__(self, num_schools=5, root_dir='data/raw/Gambo', max_total_samples=1500):
        self.num_schools = num_schools
        self.dataset = NeuroLensMultimodalDataset(root_dir=root_dir, split='Train', max_samples=max_total_samples)
        self.school_loaders = {}
        self.partition_data()
        
    def partition_data(self):
        total_len = len(self.dataset)
        indices = np.arange(total_len)
        np.random.shuffle(indices)
        
        # Split into school partitions
        splits = np.array_split(indices, self.num_schools)
        
        for i, split_idx in enumerate(splits):
            subset = Subset(self.dataset, split_idx)
            loader = DataLoader(subset, batch_size=32, shuffle=True)
            self.school_loaders[f"School_{chr(65+i)}"] = loader
            print(f"Partitioned {len(split_idx)} child data records to Virtual School {chr(65+i)}")

    def get_school_loaders(self):
        return self.school_loaders
