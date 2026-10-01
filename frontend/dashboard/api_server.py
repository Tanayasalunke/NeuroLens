import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os

from data.loaders import get_dataloader
from models.multi_task_heads import MultiModalNeuroLensModel
from models.modality_dropout import ModalityDropout
from models.cognitive_forecast import LongitudinalCognitiveForecaster
from explainability.shap_explainer import TabularFeatureImportanceExplainer
from explainability.attention_viz import CrossModalAttentionVisualizer
from explainability.llm_narrative import LLMNarrativeGenerator

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModalNeuroLensModel().to(device)

# Load trained checkpoint if available
checkpoint_path = 'checkpoints/neurolens_fusion_model.pt'
if os.path.exists(checkpoint_path):
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded trained NeuroLens fusion model from {checkpoint_path}")
    except Exception as e:
        print(f"Loaded model initialized with fresh weights ({e})")
model.eval()

mod_drop = ModalityDropout(num_modalities=5, embed_dim=64, drop_prob=0.0).to(device)
forecaster = LongitudinalCognitiveForecaster().to(device)
forecaster.eval()

shap_explainer = TabularFeatureImportanceExplainer()
attn_viz = CrossModalAttentionVisualizer()
nlg = LLMNarrativeGenerator()

@app.route('/api/child_profile', methods=['GET'])
def get_child_profile():
    loader = get_dataloader(split='Test', batch_size=1)
    batch = next(iter(loader))
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(batch_dev)
        probs = outputs['probs'][0].cpu().numpy().tolist()
        severities = outputs['severities'][0].cpu().numpy().tolist()
        
    shap_scores = shap_explainer.compute_importance_scores(model, batch_dev, target_head_idx=0)
    attn_res = attn_viz.compute_attention_weights(outputs['fused_tokens'])
    
    narrative = nlg.generate_narrative(
        child_name="Leo Parker",
        probs=probs,
        severities=severities,
        shap_scores=shap_scores,
        attn_contributions=attn_res['modality_contributions']
    )
    
    # Generate 6-month forecasted trajectory
    past_sessions = torch.tensor([[[0.82, 0.78, 0.25, 0.65],
                                   [0.74, 0.68, 0.22, 0.58],
                                   [0.61, 0.55, 0.18, 0.45],
                                   [severities[0], severities[1], severities[2], severities[3]]]], device=device)
    with torch.no_grad():
        future_trajs, efficacy = forecaster(past_sessions)
        
    f_trajs = future_trajs[0].cpu().numpy().tolist()
    eff_val = float(efficacy[0].cpu().numpy()[0])
    
    longitudinal_trajectory = [
        {"session": "Session 1 (W0)", "dyslexia": 0.82, "dysgraphia": 0.78, "dyscalculia": 0.25, "adhd": 0.65},
        {"session": "Session 2 (W4)", "dyslexia": 0.74, "dysgraphia": 0.68, "dyscalculia": 0.22, "adhd": 0.58},
        {"session": "Session 3 (W8)", "dyslexia": 0.61, "dysgraphia": 0.55, "dyscalculia": 0.18, "adhd": 0.45},
        {"session": "Session 4 (Current)", "dyslexia": round(severities[0], 2), "dysgraphia": round(severities[1], 2), "dyscalculia": round(severities[2], 2), "adhd": round(severities[3], 2)},
        {"session": "Forecast +3M", "dyslexia": round(f_trajs[0][0], 2), "dysgraphia": round(f_trajs[0][1], 2), "dyscalculia": round(f_trajs[0][2], 2), "adhd": round(f_trajs[0][3], 2)},
        {"session": "Forecast +6M", "dyslexia": round(f_trajs[1][0], 2), "dysgraphia": round(f_trajs[1][1], 2), "dyscalculia": round(f_trajs[1][2], 2), "adhd": round(f_trajs[1][3], 2)}
    ]
    
    response = {
        "child_id": "NL-2026-8842",
        "name": "Leo Parker",
        "age": 8,
        "grade": "3rd Grade",
        "risk_scores": {
            "dyslexia": round(probs[0], 3),
            "dysgraphia": round(probs[1], 3),
            "dyscalculia": round(probs[2], 3),
            "adhd_reading": round(probs[3], 3)
        },
        "severity_indices": {
            "dyslexia": round(severities[0], 3),
            "dysgraphia": round(severities[1], 3),
            "dyscalculia": round(severities[2], 3),
            "adhd_reading": round(severities[3], 3)
        },
        "shap_importance": shap_scores,
        "modality_contributions": attn_res['modality_contributions'],
        "longitudinal_trajectory": longitudinal_trajectory,
        "forecast_efficacy": round(eff_val * 100, 1),
        "llm_narrative": narrative
    }
    
    return jsonify(response)

@app.route('/api/live_predict', methods=['POST'])
def live_predict():
    """
    Real-time interactive slider prediction endpoint for dashboard sandbox demo.
    """
    data = request.json or {}
    
    # Custom slider overrides
    fixation_dur = float(data.get('fixation_duration', 650.0))
    pen_tremor = float(data.get('pen_tremor', 0.45))
    pause_ratio = float(data.get('pause_ratio', 0.38))
    eeg_tbr = float(data.get('eeg_tbr', 4.2))
    
    # Sensor availability masks
    mask_gaze = bool(data.get('has_gaze', True))
    mask_hw = bool(data.get('has_hw', True))
    mask_speech = bool(data.get('has_speech', True))
    mask_draw = bool(data.get('has_draw', True))
    mask_eeg = bool(data.get('has_eeg', True))
    
    mask_tensor = torch.tensor([[mask_gaze, mask_hw, mask_speech, mask_draw, mask_eeg]], dtype=torch.bool, device=device)
    
    loader = get_dataloader(split='Test', batch_size=1)
    batch = next(iter(loader))
    
    # Override scalar batch features with interactive slider values
    batch['gaze_feats'][0, 0] = fixation_dur / 1000.0 # Normalized fixation duration
    batch['handwriting_ts'][0, :, 3] = pen_tremor # Tremor index
    batch['speech_feats'][0, 0] = pause_ratio # Pause ratio
    batch['eeg_feats'][0, 4] = eeg_tbr / 10.0 # TBR normalized
    
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(batch_dev)
        probs = outputs['probs'][0].cpu().numpy().tolist()
        severities = outputs['severities'][0].cpu().numpy().tolist()
        
    attn_res = attn_viz.compute_attention_weights(outputs['fused_tokens'])
    
    narrative = f"Interactive Simulation Result: Given input parameters (Fixation: {fixation_dur:.0f}ms, Tremor: {pen_tremor:.2f}, Pause Ratio: {pause_ratio:.2f}, EEG TBR: {eeg_tbr:.1f}), the PyTorch multi-task fusion network projects a Dyslexia Risk of {probs[0]*100:.1f}% and Dysgraphia Severity of {severities[1]:.2f}. Active Sensors: Gaze ({mask_gaze}), HW ({mask_hw}), Speech ({mask_speech}), Draw ({mask_draw}), EEG ({mask_eeg})."
    
    return jsonify({
        "status": "success",
        "risk_scores": {
            "dyslexia": round(probs[0], 3),
            "dysgraphia": round(probs[1], 3),
            "dyscalculia": round(probs[2], 3),
            "adhd_reading": round(probs[3], 3)
        },
        "severity_indices": {
            "dyslexia": round(severities[0], 3),
            "dysgraphia": round(severities[1], 3),
            "dyscalculia": round(severities[2], 3),
            "adhd_reading": round(severities[3], 3)
        },
        "modality_contributions": attn_res['modality_contributions'],
        "live_narrative": narrative
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
