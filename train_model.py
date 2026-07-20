import os
import numpy as np
from app.feature_extractor import extract_wav2vec_features
from app.utils import preprocess_audio

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

DATASET_ROOT = "Dataset"


# -----------------------------
# Collect samples
# -----------------------------
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
                        "language": language
                    })

    return samples


# -----------------------------
# Feature extraction
# -----------------------------
def extract_wav2vec_features2(samples):
    X = []
    y = []
    skipped = 0

    for i, s in enumerate(samples):
        try:
            waveform, sr = preprocess_audio(s["path"])
            features = extract_wav2vec_features(waveform, sr)

            X.append(features)
            y.append(s["label"])
            print(f"Done {i+1}/{len(samples)}")

        except Exception as e:
            skipped += 1
            print(f"Skipping {s['path']} | Reason: {e}")

    print("Skipped files:", skipped)
    return np.array(X), np.array(y)


# -----------------------------
# Main training pipeline
# -----------------------------
samples = collect_samples()
print("Total samples:", len(samples))

X, y = extract_wav2vec_features2(samples)

print("Class distribution:")
print("Human:", np.sum(y == 0))
print("AI:", np.sum(y == 1))


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.joblib")


# -----------------------------
# PCA with variance-based selection
# -----------------------------
pca = PCA(n_components=200, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Original dim:", X_train.shape[1])
print("PCA dim:", X_train_pca.shape[1])
print("Explained variance:", pca.explained_variance_ratio_.sum())

joblib.dump(pca, "pca.joblib")

from lightgbm import LGBMClassifier

# -----------------------------
# Train LightGBM only
# -----------------------------
print("\nTraining LightGBM...")

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nLightGBM Results:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(
    y_test,
    y_pred,
    target_names=["HUMAN", "AI_GENERATED"]
))
print("Accuracy:", acc)
print("F1 Score:", f1)

# Save model
joblib.dump(model, "classifier.joblib")
print("LightGBM model saved as classifier.joblib")
