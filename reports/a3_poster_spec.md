# NeuroLens A3 Academic Research Poster Specification

## Poster Dimensions & Style Guidelines
- **Dimensions**: A3 (297mm x 420mm portrait)
- **Background Theme**: Dark Navy / Neuro-Tech (`#0A0E27`) with low-opacity cyan (`#00F5FF`) and violet (`#B026FF`) neural pathway circuit lines.
- **Typography**: Space Grotesk (Titles & Headers), Inter (Body Text).
- **Cards**: Glassmorphism translucent panels with subtle 1px cyan/violet glowing borders.

---

## 3-Column Layout Specification

### Top Banner (Full Width)
- **Title**: `NeuroLens: A Multimodal Deep Learning Framework for Early Detection and Explainable Analysis of Learning Disabilities` (Large bold, glowing cyan outline).
- **Subtitle**: *Fusing Eye-Tracking, Handwriting, Speech & EEG to Catch Learning Disabilities Before They Fall Through the Cracks*
- **Author & Affiliation**: Capstone Research Project Team, Department of Computer Science & Deep Learning.

### Column 1 — Problem & Motivation
- **Stat Callouts**:
  - `1 in 5`: Children show signs of a learning difficulty.
  - `80%`: Of cases go undetected until grade 3 due to single-modality testing limits.
- **Problem Statement**: Standard diagnostic approaches use isolated clinical forms or single-modality models. NeuroLens introduces a 5-signal cognitive-fusion paradigm.
- **Hero Motif Visual**: Child silhouette overlaid with neural wireframe emanations.

### Column 2 (Center) — Technical Architecture & Cross-Modal Fusion
- **Central Fusion Diagram**: 5 Modality Icons (Gaze, Handwriting, Speech, Drawing, EEG) converging into a central **Cross-Modal Transformer** node, branching into 4 risk gauges: Dyslexia, Dysgraphia, Dyscalculia, Attention-Deficit.
- **Pipeline Flow**: Data Loaders & Synthetic Engine → 5 Modality Encoders → Cross-Modal Transformer → Multi-Task Heads → Explainability Engine.
- **Federated Learning Block**: FedAvg across 5 Virtual Schools achieving **94.58%** global accuracy without raw data egress.

### Column 3 — Explainability, Results & Real-World Impact
- **Explainability Visuals**:
  - Grad-CAM Heatmap overlay sample on letter stroke image.
  - SHAP Horizontal Bar Chart of feature attribution drivers.
  - Chat-UI style Callout Box: *"LLM Plain-English Narrative for Parents & Teachers"*.
- **Metrics Table**:
  - Multi-Task Accuracy: **90.75%**
  - Severity Regression RMSE: **0.0867**
  - Federated FL Accuracy: **94.58%**
- **Digital Cognitive Twin Mockup**: Radar chart + Longitudinal trajectory line plot.
- **Footer**: QR code linking to demo video and repository codebase.
