# Datasheet for Synthetic Multimodal Cognitive Dataset

Following the guidelines of Gebru et al., "Datasheets for Datasets" (CACM 2021).

## Motivation
- **For what purpose was the dataset created?** Created to evaluate 5-signal biometric fusion (Gaze, Stylus, Speech, Drawing, EEG) at scale and pre-train multimodal transformer architectures before real-world fine-tuning.
- **Who created the dataset?** The NeuroLens AI Research Team.

## Composition
- **What do the instances represent?** Each instance represents a 5-signal synchronized biometric profile corresponding to a simulated child completing three interactive mini-games.
- **How many instances are there in total?** 1,500 5-signal cognitive profiles paired with 208,381 real handwriting stroke images from the Gambo dataset.
- **Does the dataset contain sensitive PII?** No. Synthetic data contains zero real child PII. Real Gambo handwriting images are fully anonymized character strokes.

## Collection & Generation Process
- **How was the data generated?** Using a Conditional GAN ($z=64, y=4$) paired with a physics-informed kinetic simulator modeling eye gaze fixations, stylus pressure curves, speech pause ratios, and EEG spectral band powers.
- **Statistical Validation**: Synthetic feature distributions were validated against real clinical literature benchmarks via two-sample Kolmogorov-Smirnov tests ($p < 0.001$).

## Uses & Limitations
- **Intended Use**: Model pre-training, architecture stress-testing, and zero-shot missing modality evaluations.
- **Limitations**: Synthetic data cannot substitute for real clinical diagnostic trials. All final accuracy numbers are evaluated on real-only and fine-tuned real datasets.
