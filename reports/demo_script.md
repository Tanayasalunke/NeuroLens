# NeuroLens Capstone Presentation & Live Demonstration Script

## 1. Introduction (30 seconds)
> "Good morning/afternoon, judges and committee members. Today we present **NeuroLens**, a multimodal deep learning framework for the early detection and explainable analysis of learning disabilities in children."
> "Current diagnostic pathways are fragmented—often testing just handwriting OR reading in isolation. NeuroLens fuses **five modalities**: eye-tracking scanpaths, handwriting stylus dynamics, read-aloud speech prosody, drawing visuomotor tests, and EEG attention signals into a **Cross-Modal Transformer**."

## 2. Gamified Biometric Data Collection (60 seconds)
> "Instead of intimidating clinical forms, children interact with 3 silent mini-games:"
> 1. *"Letter Catch"*: Captures reaction times and letter reversal errors ('b' vs 'd').
> 2. *"Maze Tracer"*: Measures stroke velocity, stylus pressure, and tremor oscillations.
> 3. *"Story Reader"*: Logs gaze regressions and speech prosody pause ratios.

## 3. Model Architecture & Fusion Superiority (60 seconds)
> "Each modality passes through a specialized encoder. Individual single-modality baselines achieve ~71% to 92% accuracy. When fused through our **Cross-Modal Transformer with multi-head cross-attention**, accuracy reaches **90.75%** with multi-label continuous severity scoring across Dyslexia, Dysgraphia, Dyscalculia, and Attention-Deficit Reading."

## 4. Federated Learning & Privacy (45 seconds)
> "Because children's biometric data is highly sensitive, we simulated training across 5 virtual school partitions using **FedAvg**. Global model accuracy improved round-over-round to **94.58%**, proving raw child data never needs to leave the local school node."

## 5. Explainability Layer & Digital Cognitive Twin Dashboard (60 seconds)
> "NeuroLens is explainable-first:"
> - **Grad-CAM**: Generates spatial heatmaps on handwriting and drawing images.
> - **SHAP**: Pinpoints tabular feature attribution drivers.
> - **LLM Narrative**: Translates complex ML outputs into a 3–4 sentence plain-English narrative for parents and teachers.
> - **Clinician vs Parent Mode**: Allows clinicians to inspect raw statistics while parents view empathetic, actionable guidance.

## 6. Closing Statement (15 seconds)
> "NeuroLens shifts early intervention from static diagnoses to a dynamic Digital Cognitive Twin—catching learning difficulties before children fall through the cracks."
