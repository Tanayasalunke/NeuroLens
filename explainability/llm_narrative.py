class LLMNarrativeGenerator:
    """
    Plain-English Natural Language Report Generator (NLG Engine) for parents & teachers.
    Translates model severity scores, SHAP values, and Grad-CAM heatmaps into empathetic,
    clear, and actionable diagnostic narratives.
    """
    def __init__(self):
        pass

    def generate_narrative(self, child_name="Child", probs=None, severities=None, shap_scores=None, attn_contributions=None):
        probs = probs or [0.75, 0.82, 0.15, 0.40]
        severities = severities or [0.78, 0.85, 0.12, 0.38]
        shap_scores = shap_scores or {}
        attn_contributions = attn_contributions or {}
        
        dyslexia_sev = severities[0]
        dysgraphia_sev = severities[1]
        dyscalculia_sev = severities[2]
        adhd_sev = severities[3]
        
        sentences = []
        
        # Sentence 1: Primary Finding
        flagged_conditions = []
        if dyslexia_sev >= 0.6:
            flagged_conditions.append("reading fluency patterns associated with Dyslexia")
        if dysgraphia_sev >= 0.6:
            flagged_conditions.append("stylus pressure and letter stroke variations characteristic of Dysgraphia")
        if dyscalculia_sev >= 0.6:
            flagged_conditions.append("visuomotor spatial processing indicators linked to Dyscalculia")
        if adhd_sev >= 0.6:
            flagged_conditions.append("gaze-fixation fluctuations indicative of attention-deficit reading patterns")
            
        if flagged_conditions:
            sentences.append(f"NeuroLens identified key indicators for {', and '.join(flagged_conditions)}.")
        else:
            sentences.append(f"NeuroLens analyzed {child_name}'s multi-modal cognitive signals and found overall balanced reading and motor patterns within typical ranges.")
            
        # Sentence 2: Modality Evidence (SHAP & Attention Driver)
        highest_modality = max(attn_contributions, key=attn_contributions.get) if attn_contributions else "Handwriting & Gaze"
        top_shap_feature = max(shap_scores, key=shap_scores.get) if shap_scores else "reading regression count"
        
        modality_details = {
            "Gaze": "longer gaze fixations and frequent backward eye movements during reading passages",
            "Handwriting": "variations in pen pressure and stroke execution speed during letter tracing",
            "Speech": "extended pause intervals and speech rate fluctuations during read-aloud activities",
            "Drawing": "minor spatial alignment shifts on drawing visual-motor tests",
            "EEG": "intermittent focus variations during reading tasks"
        }
        evidence_text = modality_details.get(highest_modality, "consistent biometrical patterns across mini-game activities")
        sentences.append(f"The assessment was primarily driven by {highest_modality} data, which highlighted {evidence_text}.")
        
        # Sentence 3: Reassurance / Motor Skill Callout
        unflagged = []
        if dysgraphia_sev < 0.4:
            unflagged.append("fine motor control and handwriting pressure were steady")
        if dyslexia_sev < 0.4:
            unflagged.append("reading gaze tracking remained fluid")
        if adhd_sev < 0.4:
            unflagged.append("attention focus levels were consistently sustained")
            
        if unflagged:
            sentences.append(f"Encouragingly, {unflagged[0]}, suggesting these cognitive areas are robust.")
            
        # Sentence 4: Recommendation / Next Steps
        sentences.append("We recommend discussing these preliminary screening insights with an educational specialist or counselor for tailored learning support.")
        
        return " ".join(sentences)
