import os, time, random, joblib, glob
import pandas as pd
import numpy as np
from scipy.signal import welch
import pygame

# ------------------ Load model ------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ Feature Extraction ------------------
bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
sf = 128

def bandpower(data, sf, band, window_sec=4, relative=True):
    freqs, psd = welch(data, sf, nperseg=int(window_sec * sf))
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bp = np.trapezoid(psd[idx_band], dx=freq_res)
    if relative:
        bp /= np.trapezoid(psd, dx=freq_res)
    return bp

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

# ------------------ Actions ------------------
pygame.mixer.init()

def play_relaxing_music():
    songs = os.listdir("relax_sounds")
    if not songs:
        print("⚠️ No relaxing sounds found!")
        return
    chosen = random.choice(songs)
    print(f"🎵 Playing: {chosen}")
    pygame.mixer.music.load(os.path.join("relax_sounds", chosen))
    pygame.mixer.music.play()

# ------------------ Agentic Analyzer ------------------
def analyze_eeg(file_path):
    df = pd.read_csv(file_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    features = extract_features(df)
    features_scaled = scaler.transform(features)
    preds = model.predict(features_scaled)

    stress_percent = np.mean(preds) * 100
    label = "Stress" if stress_percent > 50 else "Relax"

    print(f"🧠 {file_path} → Stress: {stress_percent:.2f}% → {label}")
    return label, stress_percent

def agentic_loop(folder="Live_EEG", interval=60):
    print("🤖 Agentic AI started... Monitoring EEG data.\n")
    os.makedirs(folder, exist_ok=True)

    while True:
        eeg_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if not eeg_files:
            print("No EEG files yet. Waiting...")
            time.sleep(interval)
            continue

        latest = eeg_files[-1]
        label, percent = analyze_eeg(latest)

        if label == "Stress":
            print(f"⚠️ Stress detected ({percent:.1f}%). Playing relaxing audio.")
            play_relaxing_music()
        else:
            print("✅ User relaxed. Logging state only.")

        with open("agentic_log.txt", "a") as f:
            f.write(f"{time.ctime()} - {latest} - {label} ({percent:.2f}%)\n")

        print(f"🔁 Rechecking in {interval}s...\n")
        time.sleep(interval)

# Uncomment below to start live loop
# agentic_loop("Live_EEG", interval=60)
