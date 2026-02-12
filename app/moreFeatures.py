import librosa
import numpy as np

def extract_acoustic_features(waveform, sr=16000):
    features = []

    # --- MFCC ---
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # --- Pitch ---
    pitch = librosa.yin(waveform, fmin=50, fmax=300, sr=sr)
    pitch = pitch[~np.isnan(pitch)]

    if len(pitch) > 0:
        features.append(np.mean(pitch))
        features.append(np.std(pitch))
    else:
        features.extend([0.0, 0.0])

    # --- Zero-crossing rate ---
    zcr = librosa.feature.zero_crossing_rate(waveform)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # --- Spectral centroid ---
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # --- Spectral flatness ---
    flatness = librosa.feature.spectral_flatness(y=waveform)
    features.append(np.mean(flatness))
    features.append(np.std(flatness))

    return np.array(features, dtype=np.float32)
