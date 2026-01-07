import os
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ------------------ Dataset Path ------------------
DATASET_PATH = "Audio_Speech_Actors_01-24"

# ------------------ Emotion Mapping ------------------
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ------------------ Feature Extraction ------------------
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y, _ = librosa.effects.trim(y)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    return np.hstack([mfccs, chroma, [zcr, rms]])

# ------------------ Load and Label Data ------------------
data, labels = [], []

for actor_folder in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotion_map.get(emotion_code)
            if emotion is None:
                continue

            # Assign binary stress labels
            if emotion in ["angry", "fearful", "disgust", "sad", "surprised"]:
                label = 1  # Stress
            elif emotion in ["neutral", "calm", "happy"]:
                label = 0  # Relax
            else:
                continue

            file_path = os.path.join(actor_path, file)
            features = extract_audio_features(file_path)
            data.append(features)
            labels.append(label)

print(f"✅ Loaded {len(data)} audio samples")

# ------------------ Convert to Arrays ------------------
X = np.array(data)
y = np.array(labels)

# ------------------ Scale and Split ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ------------------ Train Model ------------------
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# ------------------ Evaluate ------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------ Save Model ------------------
joblib.dump(model, "model_audio.pkl")
joblib.dump(scaler, "scaler_audio.pkl")
print("\n✅ model_audio.pkl and scaler_audio.pkl saved successfully!")
