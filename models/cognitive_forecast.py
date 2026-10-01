import torch
import torch.nn as nn
import numpy as np

class LongitudinalCognitiveForecaster(nn.Module):
    """
    Longitudinal Cognitive Drift & Therapy Trajectory Forecaster for NeuroLens.
    Models multi-session historical severity scores (e.g., Sessions 1 to 4 over 6 months)
    and predicts the projected 3-month and 6-month cognitive recovery curves,
    intervention efficacy index, and risk of persistent deficit.
    """
    def __init__(self, input_dim=4, hidden_dim=32, num_future_steps=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_future_steps = num_future_steps
        
        # Recurrent state-space predictor
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Projection heads
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_future_steps * input_dim)
        )
        
        self.therapy_efficacy_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, historical_sessions):
        """
        historical_sessions: Tensor of shape (B, num_past_sessions, 4) containing 4 disorder severities.
        Returns:
          projected_trajectories: (B, num_future_steps, 4)
          therapy_efficacy_score: (B, 1) - Estimated recovery probability (0.0 to 1.0)
        """
        B, S, D = historical_sessions.shape
        out, h_n = self.gru(historical_sessions) # h_n is (1, B, 32)
        last_hidden = h_n.squeeze(0) # (B, 32)
        
        flat_trajs = self.trajectory_head(last_hidden) # (B, future_steps * 4)
        projected_trajectories = flat_trajs.view(B, self.num_future_steps, D)
        
        # Clamp severities to valid range [0.0, 1.0]
        projected_trajectories = torch.clamp(projected_trajectories, 0.0, 1.0)
        
        therapy_efficacy = self.therapy_efficacy_head(last_hidden)
        
        return projected_trajectories, therapy_efficacy

    def generate_narrative_forecast(self, past_scores, future_scores, efficacy):
        """
        Generates clinical narrative report for Digital Cognitive Twin.
        """
        eff_val = float(efficacy.item() * 100)
        
        disorders = ["Dyslexia", "Dysgraphia", "Dyscalculia", "ADHD"]
        init_sev = np.mean(past_scores[0])
        curr_sev = np.mean(past_scores[-1])
        proj_sev = np.mean(future_scores[-1])
        
        if proj_sev < curr_sev:
            trend_str = "positive recovery trajectory with estimated " + f"{eff_val:.1f}%" + " therapy response probability."
            recommendation = "Maintain current speech-language and fine-motor intervention regimen."
        else:
            trend_str = "plateaued or deteriorating cognitive trajectory requiring regimen adjustment."
            recommendation = "Recommend clinical review to recalibrate occupational and reading therapy exercises."
            
        narrative = (
            f"Longitudinal Analysis (Sessions 1-{len(past_scores)}): Overall cognitive risk index shifted "
            f"from {init_sev:.2f} (baseline) to {curr_sev:.2f} (current session). "
            f"Forecasting models project a {trend_str} Estimated 6-month projected risk index is {proj_sev:.2f}. "
            f"Clinical Recommendation: {recommendation}"
        )
        return narrative

if __name__ == '__main__':
    forecaster = LongitudinalCognitiveForecaster()
    # Simulate 4 past sessions for 1 child
    past_sessions = torch.tensor([[[0.75, 0.80, 0.20, 0.65],
                                   [0.68, 0.72, 0.18, 0.58],
                                   [0.55, 0.60, 0.15, 0.45],
                                   [0.42, 0.48, 0.12, 0.35]]]) # Shape (1, 4, 4)
    
    future_trajs, efficacy = forecaster(past_sessions)
    narrative = forecaster.generate_narrative_forecast(past_sessions[0].numpy(), future_trajs[0].detach().numpy(), efficacy[0])
    
    print("Forecaster Execution Successful!")
    print("Projected 3-month & 6-month Severities:\n", future_trajs[0].detach().numpy())
    print("Therapy Efficacy Score:", efficacy.item())
    print("\nGenerated Clinical Narrative:\n", narrative)
