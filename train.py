import glob
import pandas as pd
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib, os

# ------------------ Load Datasets ------------------
stress_files = glob.glob("Arithmetic_CSVs/*.csv") + glob.glob("Stroop_CSVs/*.csv") + glob.glob("Mirror_CSVs/*.csv")
relax_files = glob.glob("Relax_CSVs/*.csv")

def load_data(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

stress_df = load_data(stress_files)
relax_df = load_data(relax_files)

print(f"Loaded {len(stress_files)} stress and {len(relax_files)} relax files")
print("Stress shape:", stress_df.shape)
print("Relax shape:", relax_df.shape)

# ------------------ Bandpower Feature Extraction ------------------
def bandpower(data, sf, band, window_sec=4, relative=True):
    freqs, psd = welch(data, sf, nperseg=int(window_sec * sf))
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bp = np.trapezoid(psd[idx_band], dx=freq_res)
    if relative:
        bp /= np.trapezoid(psd, dx=freq_res)
    return bp

bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
sf = 128  # sampling frequency

def extract_features(df, window_size=128*5):
    features_list = []
    for start in range(0, len(df) - window_size, window_size):
        window = df.iloc[start:start+window_size]
        features = []
        for col in df.columns:
            sig = window[col].values
            for band in bands.values():
                features.append(bandpower(sig, sf, band, relative=True))
        features_list.append(features)
    return np.array(features_list)

stress_features = extract_features(stress_df)
relax_features = extract_features(relax_df)

X = np.vstack([stress_features, relax_features])
y = np.array([1]*len(stress_features) + [0]*len(relax_features))

# ------------------ Balance + Train ------------------
X, y = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# ------------------ Evaluate ------------------
y_pred = clf.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------ Save Model ------------------
joblib.dump(clf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ Model and Scaler saved successfully!")
