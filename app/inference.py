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
    "facebook/wav2vec2-base"
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


