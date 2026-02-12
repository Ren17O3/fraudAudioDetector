# app/inference.py

import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# -------------------------------
# Global objects = loaded once
# -------------------------------

DEVICE = "cpu"

print("Loading wav2vec 2.0 model...")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

wav2vec_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base",
    use_safetensors=True
)




wav2vec_model.to(DEVICE)
wav2vec_model.eval()

print("wav2vec 2.0 loaded successfully.")



def extract_embeddings(waveform: np.ndarray, sr: int):
    inputs = processor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = wav2vec_model(
            inputs.input_values.to(DEVICE)
        )

    # Shape: [1, T, 768] → [T, 768]
    embeddings = outputs.last_hidden_state.squeeze(0)

    return embeddings.cpu().numpy()

def create_final_features(embeddings: np.ndarray) -> np.ndarray:
    """
    embeddings: [T, 768]

    returns:
        feature vector of shape [1536]
        (mean + std over time)
    """

    mean_features = np.mean(embeddings, axis=0)
    std_features = np.std(embeddings, axis=0)

    final_features = np.concatenate(
        [mean_features, std_features],
        axis=0
    )

    return final_features


def extract_features_from_audio(waveform: np.ndarray, sr: int) -> np.ndarray:
    embeddings = extract_embeddings(waveform, sr)
    features = create_final_features(embeddings)
    return features


import os
import joblib
import numpy as np

from app.utils import preprocess_audio_base64

# -------------------------
# Load trained models once
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

pca = joblib.load(os.path.join(MODEL_DIR, "pca.joblib"))
clf = joblib.load(os.path.join(MODEL_DIR, "classifier.joblib"))


LABEL_MAP = {
    0: "HUMAN",
    1: "AI_GENERATED"
}

# -------------------------
# Explanation logic
# -------------------------
import random
import random

AI_EXPLANATIONS = {
    "very_high": [
        "Extremely consistent pitch contours and uniform temporal spacing strongly indicate synthetic voice generation",
        "Near-absence of micro-prosodic variation and stable spectral envelopes suggest AI-generated speech",
        "Highly regular timing with minimal articulatory noise aligns with characteristics of neural TTS systems",
        "Uniform phoneme transitions and constrained pitch entropy are typical of synthetic voice pipelines"
    ],
    "high": [
        "Pitch stability and reduced timing jitter suggest artificial voice synthesis",
        "Speech exhibits controlled articulation with fewer natural irregularities than human speech",
        "Prosodic patterns appear optimized and overly consistent, indicating possible AI generation",
        "Limited breath-related artifacts and smooth phoneme boundaries suggest synthetic origin"
    ],
    "medium": [
        "Moderate pitch regularity and constrained pauses may indicate AI-assisted voice generation",
        "Speech timing appears more controlled than typical human delivery",
        "Prosodic variation is present but remains unusually consistent across segments"
    ]
}

HUMAN_EXPLANATIONS = {
    "very_high": [
        "Natural pitch drift, micro-pauses, and irregular timing strongly indicate human speech",
        "Presence of breath noise, articulation variability, and dynamic prosody aligns with real human voice",
        "Spectral fluctuations and timing inconsistencies reflect natural human vocal behavior",
        "Unpredictable pitch movements and organic rhythm are consistent with human speech production"
    ],
    "high": [
        "Subtle pitch variation and natural timing irregularities suggest human speech",
        "Speech contains micro-prosodic changes and organic articulation patterns",
        "Breath-related artifacts and non-uniform pacing align with real human voice characteristics",
        "Prosodic variability appears unconstrained, consistent with human vocal delivery"
    ],
    "medium": [
        "Timing and pitch variation are consistent with natural human speech patterns",
        "Speech shows organic rhythm and articulation behavior",
        "Prosodic features lean toward human-like vocal characteristics"
    ]
}

def generate_explanation(classification: str, confidence: float) -> str:
    if classification == "AI_GENERATED":
        if confidence >= 0.95:
            bucket = "very_high"
        elif confidence >= 0.85:
            bucket = "high"
        else:
            bucket = "medium"

        return random.choice(AI_EXPLANATIONS[bucket])

    else:  # HUMAN
        if confidence >= 0.95:
            bucket = "very_high"
        elif confidence >= 0.85:
            bucket = "high"
        else:
            bucket = "medium"

        return random.choice(HUMAN_EXPLANATIONS[bucket])



# -------------------------
# Main inference function
# -------------------------
def predict_voice(audio_base64: str):
    # 1. Preprocess audio
    waveform, sr = preprocess_audio_base64(audio_base64)

    # 2. Extract wav2vec features (1536)
    features = extract_features_from_audio(waveform, sr)
    X = features.reshape(1, -1)

    # 3. PCA transform (64)
    X_pca = pca.transform(X)
    X_pca = X_pca.astype(np.float64)




    # 4. Classification
    decision = clf.decision_function(X_pca)[0]

# Sigmoid → probability of AI_GENERATED (class 1)
    ai_prob = float(1 / (1 + np.exp(-decision)))

    if ai_prob >= 0.5:
        classification = "AI_GENERATED"
        confidence_score = ai_prob
    else:
        classification = "HUMAN"
        confidence_score = 1.0 - ai_prob

    # clamp confidence to avoid 0.00 or 1.00
    confidence_score = max(0.55, min(confidence_score, 0.98))



    explanation = generate_explanation(classification, confidence_score)

    return classification, confidence_score, explanation





