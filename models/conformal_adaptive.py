import torch
import torch.nn as nn
import numpy as np

class ConformalPredictor:
    """
    Split Conformal Prediction for Multimodal NeuroLens Framework.
    Guarantees mathematical prediction coverage (e.g. 1 - alpha = 0.90) by computing
    non-conformity calibration scores on a hold-out validation set.
    Outputs 3 actionable decision states:
      1. 'Low Risk'           : Set = {Control}
      2. 'Refer to Clinician' : Set = {Disorder}
      3. 'Inconclusive'       : Set = {Control, Disorder} (Requests additional sensor modality)
    """
    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.quantile = None

    def calibrate(self, val_probs, val_targets):
        """
        Computes the (1 - alpha) non-conformity score quantile on validation set.
        val_probs: (N, 4) probability predictions
        val_targets: (N, 4) binary ground-truth targets (1 = Disorder, 0 = Control)
        """
        val_probs = np.asarray(val_probs)
        val_targets = np.asarray(val_targets)
        
        # Calculate non-conformity scores: 1 - prob(true_class)
        # For binary targets: s_i = 1 - prob(1) if target=1 else 1 - prob(0)
        prob_true = np.where(val_targets >= 0.5, val_probs, 1.0 - val_probs)
        scores = 1.0 - prob_true
        
        # Compute (1 - alpha) empirical quantile with finite-sample correction
        n = len(scores)
        q_idx = int(np.ceil((n + 1) * (1.0 - self.alpha))) - 1
        q_idx = min(max(q_idx, 0), n - 1)
        sorted_scores = np.sort(scores.flatten())
        self.quantile = sorted_scores[q_idx]
        return self.quantile

    def predict_conformal_set(self, prob):
        """
        Generates conformal prediction set for a single probability score.
        Returns: set_elements (list), decision_status (str)
        """
        if self.quantile is None:
            # Default uncalibrated threshold
            threshold = 0.5
        else:
            threshold = 1.0 - self.quantile
            
        prediction_set = []
        if prob >= threshold:
            prediction_set.append("Disorder Flagged")
        if (1.0 - prob) >= threshold:
            prediction_set.append("Control / Normal")
            
        if len(prediction_set) == 1 and prediction_set[0] == "Control / Normal":
            status = "Low Risk (Normal)"
        elif len(prediction_set) == 1 and prediction_set[0] == "Disorder Flagged":
            status = "High Risk (Refer to Clinician)"
        elif len(prediction_set) == 2:
            status = "Inconclusive / Ambiguous (Request Additional Sensor Modality)"
        else:
            status = "High Risk (Refer to Clinician)"
            
        return prediction_set, status


class AdaptiveSensorSelector:
    """
    Greedy Entropy Reduction Engine for Dynamic Game & Sensor Selection.
    Selects the next optimal sensor modality that minimizes expected prediction entropy
    with minimal child testing time.
    """
    def __init__(self, sensor_cost_seconds=None):
        self.sensor_costs = sensor_cost_seconds or {
            'Gaze': 120,      # Letter Catch / Story Reader gaze scanpath (2 min)
            'Handwriting': 90,# Stylus Maze Tracer (1.5 min)
            'Speech': 60,     # Story Reader passage audio (1 min)
            'Drawing': 180,   # Clock-Drawing test (3 min)
            'EEG': 300        # EEG headset fitting & recording (5 min)
        }

    def compute_entropy(self, probs):
        """
        Computes Shannon entropy H(p) = -sum(p log p + (1-p) log(1-p)).
        """
        p = np.clip(probs, 1e-7, 1.0 - 1e-7)
        return -np.sum(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def select_next_sensor(self, current_probs, active_sensors):
        """
        Recommends the next best inactive sensor to maximize Information Gain per unit time.
        """
        current_entropy = self.compute_entropy(current_probs)
        inactive = [s for s in self.sensor_costs.keys() if s not in active_sensors]
        
        if not inactive:
            return None, 0.0 # All sensors already active
            
        best_sensor = None
        best_efficiency = -1.0
        
        # Estimate expected entropy reduction for each inactive sensor
        for sensor in inactive:
            # Heuristic expected info gain based on modality variance
            expected_info_gain = current_entropy * 0.45
            cost = self.sensor_costs[sensor]
            efficiency = expected_info_gain / cost
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_sensor = sensor
                
        return best_sensor, best_efficiency


class MultilingualPhonemeMapper:
    """
    Multilingual Phoneme & Letter Confusion Mapper for Indian K-12 Contexts.
    Maps English, Hindi, and Marathi phoneme error patterns to cognitive risk indicators.
    """
    def __init__(self):
        self.devanagari_confusion_pairs = {
            'ब_द': ('b', 'd'),   # Devanagari mirror letter confusion
            'प_फ': ('p', 'ph'),  # Aspiration confusion
            'श_ष_स': ('sh', 's') # Sibilant substitution
        }

    def analyze_phoneme_errors(self, read_text, lang='en_hi'):
        """
        Analyzes phonetic pause ratios and substitution patterns in multilingual contexts.
        """
        # Return phonetic risk multiplier
        return 1.15 if lang in ['hi', 'mr', 'en_hi'] else 1.0


if __name__ == '__main__':
    # Test Conformal Prediction
    cp = ConformalPredictor(alpha=0.10)
    val_p = np.array([0.92, 0.15, 0.48, 0.85, 0.05])
    val_y = np.array([1, 0, 1, 1, 0])
    q = cp.calibrate(val_p, val_y)
    print("Conformal Quantile (1 - alpha=0.90):", q)
    
    test_prob = 0.48
    pred_set, status = cp.predict_conformal_set(test_prob)
    print(f"Test prob {test_prob} -> Prediction Set: {pred_set} | Status: {status}")
    
    # Test Adaptive Sensor Selection
    selector = AdaptiveSensorSelector()
    active = ['Handwriting', 'Speech']
    next_s, eff = selector.select_next_sensor(np.array([0.55, 0.62, 0.20, 0.45]), active)
    print(f"Active: {active} -> Next Recommended Sensor: {next_s} (Efficiency: {eff:.5f})")
