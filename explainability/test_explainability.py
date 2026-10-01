import torch
from data.loaders import get_dataloader
from models.multi_task_heads import MultiModalNeuroLensModel
from explainability.shap_explainer import TabularFeatureImportanceExplainer
from explainability.attention_viz import CrossModalAttentionVisualizer
from explainability.llm_narrative import LLMNarrativeGenerator

def run_explainability_test():
    print("=== Testing Phase 4 Explainability Pipeline ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MultiModalNeuroLensModel().to(device)
    if torch.cuda.is_available() or True:
        try:
            model.load_state_dict(torch.load('checkpoints/neurolens_fusion_model.pt', map_location=device))
            print("Loaded trained fusion model checkpoint!")
        except Exception:
            print("Using initialized model weights for explainability test.")
            
    model.eval()
    
    # Load sample batch
    loader = get_dataloader(split='Test', batch_size=1)
    batch = next(iter(loader))
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(batch_dev)
        probs = outputs['probs'][0].cpu().numpy()
        severities = outputs['severities'][0].cpu().numpy()
        
    print(f"Predicted Probabilities: {probs.round(3)}")
    print(f"Predicted Severities   : {severities.round(3)}")
    
    # 1. SHAP Feature Importance
    shap_explainer = TabularFeatureImportanceExplainer()
    shap_scores = shap_explainer.compute_importance_scores(model, batch_dev, target_head_idx=0)
    print("\n[SHAP Feature Importances]:", shap_scores)
    
    # 2. Cross-Modal Attention Weights
    attn_viz = CrossModalAttentionVisualizer()
    attn_res = attn_viz.compute_attention_weights(outputs['fused_tokens'])
    print("\n[Modality Contributions]:", attn_res['modality_contributions'])
    
    # 3. LLM Plain-English Narrative
    nlg = LLMNarrativeGenerator()
    narrative = nlg.generate_narrative(
        child_name="Alex",
        probs=probs.tolist(),
        severities=severities.tolist(),
        shap_scores=shap_scores,
        attn_contributions=attn_res['modality_contributions']
    )
    
    print("\n[Generated LLM Plain-English Narrative]:")
    print(f"\"{narrative}\"")
    print("\n=== Phase 4 Explainability Verification Complete! ===")

if __name__ == '__main__':
    run_explainability_test()
