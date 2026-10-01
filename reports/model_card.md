# Model Card: NeuroLens Multimodal Cognitive Fusion Architecture

## Model Details
- **Model Name**: NeuroLens (Multimodal Deep Learning & Digital Cognitive Twin)
- **Model Type**: Cross-Modal Multi-Head Attention Transformer with Multi-Task Prediction Heads
- **Version**: 2.1 (Airtight Evaluation & Privacy-Utility Tradeoff Release)
- **Authors**: Anonymized for Double-Blind Review / Academic Defense
- **Framework**: PyTorch 2.x, ONNX Runtime, Flask

## Intended Use & Framing
- **Primary Intended Use**: Early screening support and explainable cognitive profile mapping for pediatricians, K-12 educators, and occupational therapists.
- **Out-of-Scope Uses**:
  - Direct or standalone clinical/medical diagnosis without licensed specialist oversight.
  - High-stakes educational placement decisions without human-in-the-loop clinical review.

## Synthetic Data & Modality Pairing Disclosure
- **Modality Pairing**: Real handwriting images from the Gambo dataset (208,381 images) are paired with synthetic 5-signal vectors generated under matching target label conditioning vector $y$. The cross-modal model learns the joint synthetic distribution conditioned on $y$, serving as a simulated multi-signal benchmark.
- **Writer-Disjoint Splitting**: Gambo images are split strictly by filename series prefix (e.g. `Reversal304`, `1_6000`), verifying zero writer overlap between train and test sets.

## Performance Metrics & Privacy-Utility Tradeoffs (Seed 42)
- **Centralized Fusion (Upper Bound)**: Accuracy = **91.80%**, Macro F1 = 0.8920, Severity RMSE = 0.0820
- **Standard FedAvg (5 Client Nodes)**: Accuracy = **90.40%**, Macro F1 = 0.8810 (-1.4% non-IID client partition penalty)
- **DP-FedAvg ($\epsilon=5.0$)**: Accuracy = **89.50%**, Macro F1 = 0.8710 (MIA attack accuracy = 58.2%)
- **DP-FedAvg ($\epsilon=2.5$)**: Accuracy = **88.65%**, Macro F1 = 0.8620 (MIA attack suppressed to **51.2%** ~ random guess)
- **Model Calibration**: Uncalibrated ECE = 0.2528; Temperature Scaled ECE (T=1.45) = **0.0380**
- **CPU Inference Latency**: 3.28 ms ± 0.16 ms per child assessment sample
- **Conformal Coverage Guarantee**: 90% guaranteed error coverage ($1-\alpha=0.90$) for the calibrated distribution.

## Limitations & Future Clinical Roadmap
1. **Simulated Signal Pairing**: Clinical validation on co-registered clinical patient recordings is required before real-world deployment.
2. **Hardware Noise**: Consumer-grade EEG headsets exhibit higher noise floors than clinical caps.
3. **Screening Support Only**: System provides screening indicators, not clinical medical diagnoses.
