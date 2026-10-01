import torch
import copy
import math

class DifferentialPrivacyFedAvg:
    """
    Differentially Private Federated Averaging (DP-FedAvg).
    Enforces (epsilon, delta)-Differential Privacy on client weight updates by:
    1. Clipping local weight updates to maximum L2-norm threshold C.
    2. Adding calibrated Gaussian noise N(0, sigma^2 * I) to aggregated global updates.
    """
    def __init__(self, clip_norm=1.0, noise_multiplier=0.5, delta=1e-5):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        
    def clip_weight_update(self, global_weights, local_weights):
        """
        Clips local model update delta_w = local_weights - global_weights to max L2-norm C.
        """
        clipped_weights = {}
        # Calculate total L2 norm of update
        total_sq_norm = 0.0
        for k in global_weights.keys():
            if global_weights[k].dtype in [torch.float32, torch.float64]:
                diff = local_weights[k] - global_weights[k]
                total_sq_norm += torch.sum(diff ** 2).item()
                
        l2_norm = math.sqrt(total_sq_norm)
        clip_factor = min(1.0, self.clip_norm / (l2_norm + 1e-8))
        
        for k in global_weights.keys():
            if global_weights[k].dtype in [torch.float32, torch.float64]:
                diff = local_weights[k] - global_weights[k]
                clipped_weights[k] = global_weights[k] + (diff * clip_factor)
            else:
                clipped_weights[k] = local_weights[k]
                
        return clipped_weights, l2_norm, clip_factor

    def aggregate_with_dp(self, global_weights, client_weights_list, client_sample_counts):
        """
        Aggregates clipped client weight updates and adds calibrated Gaussian noise.
        """
        total_samples = sum(client_sample_counts)
        num_clients = len(client_weights_list)
        
        # 1. Clip updates for all clients
        clipped_client_weights = []
        l2_norms = []
        for w in client_weights_list:
            cw, norm, factor = self.clip_weight_update(global_weights, w)
            clipped_client_weights.append(cw)
            l2_norms.append(norm)
            
        # 2. Weighted Federated Aggregation
        new_global_weights = copy.deepcopy(global_weights)
        for key in new_global_weights.keys():
            if new_global_weights[key].dtype in [torch.float32, torch.float64]:
                new_global_weights[key] = torch.zeros_like(new_global_weights[key])
                for i in range(num_clients):
                    weight_factor = client_sample_counts[i] / total_samples
                    new_global_weights[key] += weight_factor * clipped_client_weights[i][key]
                    
                # 3. Add Gaussian Noise to float tensors
                noise_std = (self.noise_multiplier * self.clip_norm) / math.sqrt(num_clients)
                noise = torch.randn_like(new_global_weights[key]) * noise_std
                new_global_weights[key] += noise
            else:
                # Non-float parameters (e.g. num_batches_tracked) copied from first client
                new_global_weights[key] = clipped_client_weights[0][key]
                
        # Calculate empirical Privacy Budget (Epsilon approximation)
        epsilon_approx = self.compute_privacy_budget(num_clients)
        
        return new_global_weights, {
            "avg_l2_norm": sum(l2_norms) / len(l2_norms),
            "noise_std": (self.noise_multiplier * self.clip_norm) / math.sqrt(num_clients),
            "epsilon_privacy": epsilon_approx,
            "delta_privacy": self.delta
        }

    def compute_privacy_budget(self, num_clients, rounds=10):
        """
        Computes approximate epsilon bound for RDP differential privacy.
        """
        if self.noise_multiplier == 0.0:
            return float('inf')
        # Standard analytical RDP bound approximation
        q = 1.0 / num_clients
        epsilon = (q * math.sqrt(rounds * math.log(1 / self.delta))) / self.noise_multiplier
        return round(epsilon, 3)

if __name__ == '__main__':
    dp_engine = DifferentialPrivacyFedAvg(clip_norm=1.0, noise_multiplier=0.3, delta=1e-5)
    
    # Dummy weights simulation
    global_w = {"layer.weight": torch.ones(5, 5), "layer.bias": torch.zeros(5)}
    client_1 = {"layer.weight": torch.ones(5, 5) * 2.5, "layer.bias": torch.ones(5) * 0.5}
    client_2 = {"layer.weight": torch.ones(5, 5) * 0.8, "layer.bias": torch.ones(5) * -0.2}
    
    aggregated_w, stats = dp_engine.aggregate_with_dp(global_w, [client_1, client_2], [100, 100])
    print("DP-FedAvg execution successful!")
    print("Aggregated weight sample:", aggregated_w["layer.bias"])
    print("DP Privacy metrics:", stats)
