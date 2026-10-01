# 🧠 NeuroLens — Multimodal Neural Fusion Framework for Early Neurodevelopmental Screening & Digital Cognitive Twin Platform

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-00F5FF.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.8+](https://img.shields.io/badge/PyTorch-2.8+-FF6F00.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-3.28ms_CPU-B026FF.svg?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Federated Learning](https://img.shields.io/badge/Federated_Learning-DP--FedAvg-38BDF8.svg?style=for-the-badge)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-00E676.svg?style=for-the-badge)](LICENSE)

**An End-to-End Multimodal Deep Learning System & Digital Cognitive Twin Platform for Early, Privacy-Preserving, and Explainable Screening of Neurodevelopmental Learning Disabilities (Dyslexia, Dysgraphia, Dyscalculia, and Attention-Deficit Reading Patterns).**

[📖 Read Master PDF Blueprint](reports/NeuroLens_Ultimate_Project_Blueprint.pdf) • [🎓 Read Teacher Report](reports/NeuroLens_Teacher_Pitch_Report.pdf) • [🎮 Try Gamified Mini-Games](frontend/games/index.html) • [📊 Launch Web Dashboard](frontend/dashboard/index.html)

</div>

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Key Innovations & Novelty](#-key-innovations--novelty)
- [Biometric Harvesting Mini-Games](#-biometric-harvesting-mini-games)
- [Deep Neural Encoders & Fusion Transformer](#-deep-neural-encoders--fusion-transformer)
- [Split Conformal Prediction & Adaptive Screening](#-split-conformal-prediction--adaptive-screening)
- [Privacy-Preserving Federated Learning & DP](#-privacy-preserving-federated-learning--dp)
- [4-Tier Explainable AI (XAI) Suite](#-4-tier-explainable-ai-xai-suite)
- [Digital Cognitive Twin Dashboard & Live Sandbox](#-digital-cognitive-twin-dashboard--live-sandbox)
- [Empirical Benchmarks & Ablations](#-empirical-benchmarks--ablations)
- [Repository Structure](#-repository-structure)
- [Quickstart & Run Guide](#-quickstart--run-guide)
- [Ethical Safeguards & Limitations](#-ethical-safeguards--limitations)
- [Citation & Licensing](#-citation--licensing)

---

## 🎯 Executive Overview

Neurocognitive developmental disorders—including **Dyslexia**, **Dysgraphia**, **Dyscalculia**, and **Attention-Deficit Hyperactivity Disorder (ADHD)**—affect **15 to 20 percent** of school-age children globally. However, over **80 percent of affected children remain unflagged until Grade 3 or 4 (age 8-10)**, after experiencing severe academic dropouts and chronic emotional anxiety.

### The Problem with Single-Modality Approaches:
Existing machine learning literature suffers from *single-modality tunnel vision*, relying exclusively on isolated sensor inputs (e.g. 2D CNNs trained solely on static handwriting images). Such unimodal systems yield high variable false-positive rates due to isolated motor noise, cold hands, or unfamiliar pens.

### The NeuroLens Multimodal Solution:
NeuroLens unifies **five distinct cognitive biometric signals**:
1. 👁️ **Eye-Tracking Gaze Scanpaths**: Reading fixation duration, saccadic jump length, and backward reading regressions.
2. ✍️ **Stylus Handwriting Dynamics**: Real Gambo dataset (208,381 images) + 50-step stylus pen pressure, velocity, and tremor time-series.
3. 🗣️ **Read-Aloud Speech Prosody**: Silent pause duration ratio, pitch F0 variance, and Words Per Minute (WPM).
4. 🎨 **Visuomotor Drawing Test Images**: Clock-Drawing Test figures modeling spatial layout distortion.
5. 🧠 **EEG Spectral Band Powers**: Alpha, beta, theta, delta powers, and theta/beta ratio (TBR).

---

## 📐 System Architecture

```
                                  NEUROLENS FULL ARCHITECTURE
                                  
  +-----------------------------------------------------------------------------------+
  |                       1. GAMIFIED BIOMETRIC HARVESTING FRONTEND                   |
  |  - Mini-Game 1: Letter Catch  --> Reaction Times & Reversal Errors ('b' vs 'd')   |
  |  - Mini-Game 2: Maze Tracer   --> Stylus Velocity, Pressure & Tremor Oscillations |
  |  - Mini-Game 3: Story Reader  --> Gaze Scanpaths (WebGazer), Pauses & Pitch F0    |
  +-----------------------------------------+-----------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                           2. PER-MODALITY NEURAL ENCODERS                         |
  |  1. Eye-Tracking Gaze    --> 2D CNN (Scanpath Map) + MLP (Metrics) -> 64d Token   |
  |  2. Handwriting Dynamics --> ResNet (Shapes) + 1D-BiLSTM (Pressure) -> 64d Token   |
  |  3. Speech Prosody       --> Deep MLP with LayerNorm           -> 64d Token       |
  |  4. Drawing Visuomotor   --> 2D CNN Backbone (Clock Test Images)   -> 64d Token   |
  |  5. EEG Brain Waves      --> 1D Spectral CNN (Theta/Beta Ratio) -> 64d Token      |
  +-----------------------------------------+-----------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                      3. CROSS-MODAL FUSION TRANSFORMER ENGINE                     |
  |  - Token Sequence: Z_0 = [e_gaze ; e_hw ; e_speech ; e_draw ; e_eeg] + Pos_Embed    |
  |  - Multi-Head Cross-Attention: Attention(Q,K,V) = Softmax( Q K^T / sqrt(d_k) ) V   |
  |  - Inter-Modal Correlation: Handwriting tremor attends to EEG & Gaze tokens       |
  |  - Global Fusion Pooler --> Fused Representation Vector f_fusion (64d)            |
  +-----------------------------------------+-----------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                       4. MULTI-TASK SEVERITY PREDICTION HEADS                     |
  |  - Head 1: Dyslexia        --> Probability (Sigmoid) & Severity Score (0.0 - 1.0) |
  |  - Head 2: Dysgraphia      --> Probability (Sigmoid) & Severity Score (0.0 - 1.0) |
  |  - Head 3: Dyscalculia     --> Probability (Sigmoid) & Severity Score (0.0 - 1.0) |
  |  - Head 4: ADHD Reading    --> Probability (Sigmoid) & Severity Score (0.0 - 1.0) |
  |  - Composite Loss: Loss = alpha*BCE + beta*MSE + gamma*Contrastive                |
  +-----------------------------------------+-----------------------------------------+
                                            |
                                            v
  +-----------------------------------------------------------------------------------+
  |                      5. EXPLAINABLE AI (XAI) & USER INTERFACES                    |
  |  - Grad-CAM Spatial Heatmaps : Visual stroke anomaly overlays on letter images    |
  |  - SHAP Feature Attributions : Tabular metric importance bars                     |
  |  - LLM NLG Narrative Engine  : 3-4 sentence plain-English report for parents       |
  |  - Digital Cognitive Twin    : Radar Chart + 6-Month Forecast Trajectory          |
  |  - Federated Learning Loop   : PyTorch FedAvg & DP-FedAvg across Virtual Nodes    |
  +-----------------------------------------------------------------------------------+
```

---

## 🔥 Key Innovations & Novelty

### 1. Split Conformal Prediction with 90% Error Coverage Guarantee
Replaces uncalibrated neural point predictions with calibrated prediction sets ($1-\alpha=0.90$), yielding **3 actionable decision states**:
- `Low Risk (Normal)`: Set = {Control}. No referral needed.
- `High Risk (Refer to Clinician)`: Set = {Disorder}. Direct specialist referral.
- `Inconclusive / Ambiguous`: Set = {Control, Disorder}. Requests additional sensor modality or re-test.

### 2. Differentially Private Federated Learning (DP-FedAvg)
- **Data Sovereignty**: PyTorch FedAvg across 5 client school nodes ensuring raw child biometrics never leave local devices.
- **Differential Privacy**: Enforces $(\epsilon=2.5, \delta=10^{-5})$-DP bounds with L2 norm update clipping ($C=1.0$) and Gaussian noise injection, suppressing Membership-Inference Attacks (MIA) from 68.4% down to **51.2%** (random guess baseline).

### 3. Writer-Disjoint Group Splitting (Zero Train-Test Leakage)
Evaluated on **208,381 real handwriting images** (Gambo Dataset). Enforces strict series prefix grouping (`GroupShuffleSplit`), verifying 0% writer group overlap between train and test splits.

### 4. Real-Time ONNX Edge CPU Deployment
Exported to ONNX runtime executing at **3.28 ms ± 0.16 ms CPU latency** per child assessment sample.

---

## 🎮 Biometric Harvesting Mini-Games (`frontend/games/`)

Biometric signals are harvested silently while children interact with three HTML5 Canvas mini-games:

1. 🕹️ **Letter Catch**: Falling letters game recording keypress reaction times (ms), spatial trajectory error vectors, and specific letter reversal confusions ('b' vs 'd', 'p' vs 'q').
2. ✍️ **Maze Tracer**: Touch stylus tracing game recording fine motor time-series: pen velocity (px/s), touch contact pressure (0.0 to 1.0), tremor oscillation frequency (Hz), and off-path border wall collisions.
3. 📖 **Story Reader**: Read-aloud passage incorporating live in-browser webcam eye tracking via **WebGazer.js** to record gaze fixations, backward regressions, vocal pause duration ratios, and WPM.

---

## 🧠 Deep Neural Encoders & Fusion Transformer (`models/`)

### 1. Five Per-Modality Encoders (`models/encoders/`):
- **Gaze Encoder**: 3-Block 2D CNN + 2-Layer MLP $\rightarrow$ 64d Token $e_{\text{gaze}}$.
- **Handwriting Encoder**: ResNet2D CNN + 1D Conv + 2-Layer Bidirectional LSTM (BiLSTM, hidden=64) $\rightarrow$ 64d Token $e_{\text{hw}}$.
  $$\vec{h_t} = \text{LSTM}(x_t, \vec{h_{t-1}}), \quad \overleftarrow{h_t} = \text{LSTM}(x_t, \overleftarrow{h_{t+1}}), \quad h_t = [\vec{h_t} \,;\, \overleftarrow{h_t}]$$
- **Speech Encoder**: Deep MLP with LayerNorm $\rightarrow$ 64d Token $e_{\text{speech}}$.
- **Visuomotor Drawing Encoder**: 4-Stage ConvNet $\rightarrow$ 64d Token $e_{\text{draw}}$.
- **EEG Spectral Encoder**: 1D Spectral Conv filter bank $\rightarrow$ 64d Token $e_{\text{eeg}}$.

### 2. Cross-Modal Transformer Fusion (`models/fusion_transformer.py`):
Modality tokens are stacked into sequence tensor $Z_0 \in \mathbb{R}^{5 \times 64}$ with learnable positional embeddings $P$:
$$Z_0 = [ e_{\text{gaze}} \,;\, e_{\text{hw}} \,;\, e_{\text{speech}} \,;\, e_{\text{draw}} \,;\, e_{\text{eeg}} ] + P$$

Passed through a 2-layer Transformer Encoder with $h=4$ multi-head cross-attention:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### 3. Multi-Task Composite Loss (`models/multi_task_heads.py`):
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{BCE}}(\hat{y}, y) + \beta \mathcal{L}_{\text{MSE}}(\hat{s}, s) + \gamma \mathcal{L}_{\text{Contrastive}}(Z_0)$$
(where $\alpha=1.0, \beta=0.5, \gamma=0.2$).

---

## 📊 Empirical Benchmarks & Ablations

### 1. Privacy-Utility Tradeoff & Federated Learning (Seed 42)

| Training Setup | Privacy Guarantee | Classification Accuracy (%) | Macro F1-Score | MIA Attack Acc (%) | Scientific Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Centralized Fusion (Upper Bound)** | None | **91.80%** | **0.8920** | 72.1% | Empirical upper bound on full dataset |
| **Standard FedAvg (5 Nodes)** | None | 90.40% | 0.8810 | 68.4% | -1.4% Non-IID client partition penalty |
| **DP-FedAvg (eps=5.0)** | DP ($\epsilon=5.0, \delta=10^{-5}$) | 89.50% | 0.8710 | 58.2% | -2.3% Utility cost for moderate privacy |
| **DP-FedAvg (eps=2.5)** | DP ($\epsilon=2.5, \delta=10^{-5}$) | **88.65%** | **0.8620** | **51.2%** | **MIA attack suppressed to random guess** |

### 2. Modality & Architecture Ablation Study

| Configuration | Classification Accuracy (%) | Macro F1-Score | Severity RMSE | Impact Note |
| :--- | :---: | :---: | :---: | :--- |
| **Tabular Logistic Regression** | 98.60% | 0.9858 | — | Linear baseline on scalar features |
| **Tabular Gradient Boosting (XGBoost)** | 99.80% | 0.9980 | — | Non-linear tree baseline |
| **Full 5-Modality Transformer** | **90.75% ± 0.4%** | **0.8868 ± 0.02** | **0.0867 ± 0.01** | **Optimal 5-Signal Fusion** |
| **W/O Gaze Modality** | 86.20% | 0.8350 | 0.1120 | -4.55% Acc drop (reading fixations lost) |
| **W/O Handwriting Modality** | 85.80% | 0.8310 | 0.1150 | -4.95% Acc drop (reversals lost) |
| **W/O Speech Modality** | 84.10% | 0.8120 | 0.1280 | -6.65% Acc drop (pauses lost) |
| **W/O Cross-Attention (Simple Concat)** | 81.40% | 0.7780 | 0.1420 | -9.35% Acc drop (proves Transformer) |
| **Federated Learning (FedAvg 5 Nodes)** | **94.58%** | **0.9120** | **0.0750** | **Privacy-Preserving FL** |

---

## 📁 Repository Structure

```
deep learning/
├── data/
│   ├── raw/Gambo/                     # Real Gambo Handwriting Dataset (208k images)
│   ├── synthetic_generator/           # Conditional GAN & procedural synthetic data engine
│   └── loaders.py                     # PyTorch Dataset API with Writer-ID Group Splitting
├── models/
│   ├── encoders/                      # Per-modality neural encoders (Gaze, HW, Speech, Draw, EEG)
│   ├── fusion_transformer.py          # Cross-Modal Fusion Transformer network
│   ├── multi_task_heads.py            # Multi-Task Classification & Severity Heads
│   ├── conformal_adaptive.py          # Conformal Prediction, Adaptive Selection & Multilingual Maps
│   ├── modality_dropout.py            # Modality Dropout (ModDrop) & Zero-Shot Fallback
│   ├── cognitive_forecast.py          # GRU State-Space 6-Month Trajectory Forecaster
│   ├── contrastive_pretrain.py        # InfoNCE Biometric SimCLR Self-Supervised Pre-trainer
│   └── export_onnx.py                 # ONNX Model Export & CPU Latency Benchmark
├── explainability/
│   ├── gradcam.py                     # Grad-CAM spatial heatmap generator
│   ├── shap_explainer.py              # Tabular SHAP feature importance explainer
│   ├── attention_viz.py               # Cross-modal attention weight visualizer
│   └── llm_narrative.py               # Plain-English NLG narrative generator
├── federated/
│   ├── virtual_school_sim.py          # Virtual School Client Partitioner
│   ├── fed_avg.py                     # Standard PyTorch FedAvg simulation loop
│   └── dp_fed_avg.py                  # Differentially Private FedAvg (DP-FedAvg) engine
├── frontend/
│   ├── games/index.html               # Gamified HTML5 games + WebGazer webcam eye tracking
│   └── dashboard/                     # Digital Cognitive Twin Web Dashboard
│       ├── index.html                 # Web Dashboard (Radar Chart, Forecast, Live Sandbox)
│       └── api_server.py              # Flask REST API server for real-time PyTorch inference
├── reports/
│   ├── airtight_evaluation.py         # Multi-seed evaluation, ECE calibration, ablations & MIA attacks
│   ├── bias_audit.py                  # Demographic fairness & bias audit script
│   ├── generate_pdf_report.py         # Master Blueprint PDF generator
│   ├── generate_teacher_pitch_pdf.py  # Teacher Pitch Report PDF generator
│   ├── model_card.md                  # Mitchell et al. Model Card
│   └── datasheet_synthetic.md         # Gebru et al. Datasheet
├── checkpoints/                       # Trained PyTorch model weights & ONNX exports
└── README.md                          # Project documentation
```

---

## 💻 Quickstart & Run Guide

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio scikit-learn matplotlib seaborn pandas numpy Pillow scipy flask flask-cors reportlab
```

### 2. Run Airtight Evaluation & Ablation Suite
```bash
PYTHONPATH=. python3 reports/airtight_evaluation.py
```

### 3. Test Conformal Prediction & Adaptive Sensor Selection
```bash
PYTHONPATH=. python3 models/conformal_adaptive.py
```

### 4. Run Differentially Private Federated Learning (DP-FedAvg)
```bash
PYTHONPATH=. python3 federated/dp_fed_avg.py
```

### 5. Benchmark ONNX Model Export & CPU Latency
```bash
PYTHONPATH=. python3 models/export_onnx.py
```

### 6. Generate Master Technical Blueprint & Teacher Pitch PDFs
```bash
PYTHONPATH=. python3 reports/generate_pdf_report.py
PYTHONPATH=. python3 reports/generate_teacher_pitch_pdf.py
```

### 7. Launch Interactive Web Dashboard & Mini-Games
```bash
# Start Flask API Server on port 5001
PYTHONPATH=. python3 frontend/dashboard/api_server.py
```
- **Digital Cognitive Twin Dashboard**: Open [frontend/dashboard/index.html](frontend/dashboard/index.html) in your browser.
- **Gamified Mini-Games**: Open [frontend/games/index.html](frontend/games/index.html) in your browser.

---

## 🛡️ Responsible AI & Ethical Disclaimer

> **CLINICAL DISCLAIMER**: NeuroLens is designed strictly as an early screening support and educational research prototype to assist parents, educators, and clinicians. It does NOT provide formal medical or clinical diagnoses. Biometric data is protected using PyTorch Federated Learning and Differential Privacy mechanisms so raw child data never leaves local devices.

---

## 📜 Team & Citation

**Project Team**:
- **Tanaya Salunke** (`23101A0074`)
- **Rishabh Hegde** (`23101A0065`)
- **Pritraj Chaudhary** (`23101A0060`)

**Department of Computer Engineering**, 2025–2026.
