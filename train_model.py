import os
import io
import soundfile as sf
import numpy as np
import librosa
from app.inference import extract_features_from_audio

DATASET_ROOT = "Dataset"

def collect_samples():
    samples = []

    for label_name, label_value in [("Human", 0), ("AI", 1)]:
        class_dir = os.path.join(DATASET_ROOT, label_name)

        for language in os.listdir(class_dir):
            lang_dir = os.path.join(class_dir, language)

            for fname in os.listdir(lang_dir):
                if fname.endswith(".wav") or fname.endswith(".mp3"):
                    samples.append({
                        "path": os.path.join(lang_dir, fname),
                        "label": label_value,
                        "language": language,
                        "ext": os.path.splitext(fname)[1].lower()
                    })

    return samples




def preprocess_audio_for_training(path: str, target_sr=16000):
    waveform, sr = sf.read(path)

    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)

    if sr != target_sr:
        waveform = librosa.resample(
    y=waveform,
    orig_sr=sr,
    target_sr=target_sr
)


    return waveform.astype(np.float32), target_sr





def extract_wav2vec_features(samples):
    X = []
    y = []
    skipped = 0
    no = 0
    for s in samples:
        
        try:
            waveform, sr = preprocess_audio_for_training(s["path"])
            features = extract_features_from_audio(waveform, sr)

            X.append(features)
            y.append(s["label"])
            no = no+1
            print(f"Done {no}")

        except Exception as e:
            skipped += 1
            print(f"Skipping {s['path']} | Reason: {e}")

    print("Skipped files:", skipped)
    return np.array(X), np.array(y)


from sklearn.model_selection import train_test_split

samples = collect_samples()
X, y = extract_wav2vec_features(samples)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.decomposition import PCA

pca = PCA(
    n_components=64,
    random_state=42
)

X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

print("Original dim:", X_train.shape[1])
print("PCA dim:", X_train_pca.shape[1])
print("Explained variance:", pca.explained_variance_ratio_.sum())

import joblib
joblib.dump(pca, "pca.joblib")

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    solver="liblinear",
    max_iter=1000
)

clf.fit(X_train_pca, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf.predict(X_test_pca)

print(confusion_matrix(y_test, y_pred))
print(classification_report(
    y_test,
    y_pred,
    target_names=["HUMAN", "AI_GENERATED"]
))

joblib.dump(clf, "classifier.joblib")



