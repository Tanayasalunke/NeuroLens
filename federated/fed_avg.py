import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score

from data.loaders import get_dataloader
from models.multi_task_heads import MultiModalNeuroLensModel
from federated.virtual_school_sim import VirtualSchoolPartitioner

def federated_averaging(global_model, client_models, client_weights):
    """FedAvg Server Aggregation Step: Computes weighted average of client model parameters."""
    global_dict = global_model.state_dict()
    total_samples = sum(client_weights.values())
    
    for k in global_dict.keys():
        weighted_sum = torch.zeros_like(global_dict[k], dtype=torch.float32)
        for school_name, client_model in client_models.items():
            weight = client_weights[school_name] / total_samples
            weighted_sum += client_model.state_dict()[k].to(global_dict[k].device) * weight
        global_dict[k] = weighted_sum
        
    global_model.load_state_dict(global_dict)
    return global_model

def evaluate_fl_global_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch_dev)
            probs = outputs['probs'].cpu().numpy()
            targets = (batch_dev['labels'].cpu().numpy() >= 0.5).astype(np.float32)
            all_preds.append((probs >= 0.5).astype(np.float32))
            all_targets.append(targets)
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    return float(accuracy_score(all_targets.flatten(), all_preds.flatten()))

def run_federated_learning_simulation(fl_rounds=5, local_epochs=1):
    print("=== Phase 5: Federated Learning (FedAvg) Simulation Across Virtual Schools ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    partitioner = VirtualSchoolPartitioner(num_schools=5, max_total_samples=1200)
    school_loaders = partitioner.get_school_loaders()
    test_loader = get_dataloader(split='Test', batch_size=32, max_samples=300)
    
    global_model = MultiModalNeuroLensModel().to(device)
    criterion = nn.BCELoss()
    
    fl_accuracies = []
    # Centralized baseline comparison metric (~90.75%)
    centralized_baseline_acc = 90.75
    
    print(f"\nInitiating FedAvg Training across 5 Virtual School Nodes ({fl_rounds} Communication Rounds)...")
    
    for r in range(fl_rounds):
        client_models = {}
        client_weights = {}
        
        for school_name, loader in school_loaders.items():
            # Broadcast global model to virtual school client node
            client_model = copy.deepcopy(global_model).to(device)
            client_model.train()
            optimizer = optim.Adam(client_model.parameters(), lr=0.001)
            
            # Local training on private virtual school partition
            for _ in range(local_epochs):
                for batch in loader:
                    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    targets = (batch_dev['labels'] >= 0.5).float()
                    optimizer.zero_grad()
                    outputs = client_model(batch_dev)
                    loss = criterion(outputs['probs'], targets)
                    loss.backward()
                    optimizer.step()
                    
            client_models[school_name] = client_model
            client_weights[school_name] = len(loader.dataset)
            
        # Server Aggregation via FedAvg
        global_model = federated_averaging(global_model, client_models, client_weights)
        
        # Evaluate Round Global Model
        round_acc = evaluate_fl_global_model(global_model, test_loader, device) * 100
        fl_accuracies.append(round_acc)
        print(f"FL Round [{r+1}/{fl_rounds}] Global Federated Accuracy: {round_acc:6.2f}%")
        
    # 4. Plot Federated vs Centralized Privacy-Performance Tradeoff
    os.makedirs('federated', exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=300)
    rounds_x = list(range(1, fl_rounds + 1))
    
    plt.plot(rounds_x, fl_accuracies, marker='o', linewidth=2.5, color='#00F5FF', label='Federated Learning (FedAvg 5 Schools)')
    plt.axhline(y=centralized_baseline_acc, color='#FF007F', linestyle='--', linewidth=2, label=f'Centralized Baseline ({centralized_baseline_acc:.2f}%)')
    
    plt.title("NeuroLens Phase 5: Federated Learning vs Centralized Accuracy", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Federated Communication Round", fontsize=12)
    plt.ylabel("Multi-Task Classification Accuracy (%)", fontsize=12)
    plt.ylim(60, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = "federated/fl_privacy_performance_tradeoff.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved FL privacy vs performance comparison plot to {plot_path}")
    print(f"Final FL Global Accuracy after {fl_rounds} rounds: {fl_accuracies[-1]:.2f}% (vs Centralized {centralized_baseline_acc:.2f}%)")
    print("=== Phase 5 Execution Complete! ===")

if __name__ == '__main__':
    run_federated_learning_simulation(fl_rounds=5)
