import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.signal import welch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="🧠 Multimodal Agentic Stress Analyzer", layout="wide")
st.markdown("""
    <style>
        body { background-color: #0e1117; color: white; }
        .title { text-align:center; color:#76FF03; font-size:2rem; font-weight:bold; }
        .metric { font-size:1.6rem; font-weight:bold; }
    </style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODELS ------------------
model_eeg = joblib.load("model.pkl")
scaler_eeg = joblib.load("scaler.pkl")
model_audio = joblib.load("model_audio.pkl")
scaler_audio = joblib.load("scaler_audio.pkl")

# ------------------ DSP UTILS (EEG) ------------------
bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
sf = 128

def bandpower(data, sf, band, window_sec=4, relative=True):
    freqs, psd = welch(data, sf, nperseg=int(window_sec * sf))
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bp = np.trapezoid(psd[idx_band], dx=freq_res)
    if relative:
        bp /= np.trapezoid(psd, dx=freq_res)
    return bp

def extract_eeg_features(df, window_size=128*5):
    features_list = []
    for start in range(0, len(df) - window_size, window_size):
        window = df.iloc[start:start+window_size]
        features = []
        for col in df.columns:
            sig = window[col].values
            for band in bands.values():
                features.append(bandpower(sig, sf, band))
        features_list.append(features)
    return np.array(features_list)

# ------------------ AUDIO FEATURE UTILS ------------------
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y, _ = librosa.effects.trim(y)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    return np.hstack([mfccs, chroma, [zcr, rms]])

# ------------------ AGENTIC RECOMMENDER ------------------
def recommend_activities(stress_level):
    """Return recommended activities based on stress level."""
    if stress_level <= 40:
        return {
            "status": "🟢 You’re calm and balanced!",
            "recommendations": [
                "Take a short mindful walk 🌿",
                "Listen to your favorite upbeat song 🎵",
                "Drink some water 💧",
                "Reflect on one thing you’re grateful for 💭"
            ]
        }
    elif 40 < stress_level <= 70:
        return {
            "status": "🟠 Moderate stress detected.",
            "recommendations": [
                "Try a 5-minute deep breathing exercise 🧘‍♂️",
                "Step away from screens for 10 minutes 💻🚫",
                "Listen to a calming lo-fi track 🎧",
                "Stretch your arms and shoulders 🏋️‍♂️"
            ]
        }
    else:
        return {
            "status": "🔴 High stress detected! Immediate relaxation recommended.",
            "recommendations": [
                "Start guided meditation (Headspace/YouTube) 🕊️",
                "Play the provided relaxation audio 🎵",
                "Close your eyes and breathe deeply for 3 minutes 🫁",
                "Avoid caffeine or heavy work for 30 minutes ☕🚫"
            ]
        }

# ------------------ HEADER ------------------
st.markdown("<p class='title'>🧠 Agentic Multimodal Stress Analyzer</p>", unsafe_allow_html=True)
st.caption("EEG + Acoustic Signal Stress Detection | Developed by Jairam | DSP + AI + Streamlit")

tab1, tab2 = st.tabs(["📊 EEG Analysis", "🎙️ Acoustic Analysis"])

# ------------------ SESSION STATE ------------------
if "eeg_score" not in st.session_state:
    st.session_state.eeg_score = None
if "audio_score" not in st.session_state:
    st.session_state.audio_score = None
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ TAB 1: EEG ------------------
with tab1:
    uploaded_eeg = st.file_uploader("📂 Upload EEG CSV file", type=["csv"])
    if uploaded_eeg:
        df = pd.read_csv(uploaded_eeg)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        st.success("✅ EEG data uploaded successfully!")
        st.dataframe(df.head(5))

        if st.button("🔍 Analyze EEG Stress (Save for Fusion)"):
            features = extract_eeg_features(df)
            scaled = scaler_eeg.transform(features)
            preds = model_eeg.predict_proba(scaled)[:, 1]
            eeg_score = np.mean(preds) * 100
            st.session_state.eeg_score = eeg_score

            label = "😣 High Stress" if eeg_score > 50 else "😌 Relaxed"
            color = "red" if eeg_score > 50 else "green"

            st.markdown(f"""
            <div style="background-color:{color};padding:15px;border-radius:10px;">
                <h3 style="color:white;">{label}</h3>
                <p style="color:white;">Stress Probability: <b>{eeg_score:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.metric("EEG Stress Level", f"{eeg_score:.2f}%")
            st.metric("Prediction", label)
            st.success(f"🧠 EEG stress score saved: {eeg_score:.2f}%")

# ------------------ TAB 2: ACOUSTIC ------------------
with tab2:
    uploaded_audio = st.file_uploader("🎧 Upload Audio File (WAV)", type=["wav"])
    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/wav")
        if st.button("🎤 Analyze Acoustic Stress (Save for Fusion)"):
            with open("temp.wav", "wb") as f:
                f.write(uploaded_audio.read())
            features = extract_audio_features("temp.wav")
            features_scaled = scaler_audio.transform([features])
            prob = model_audio.predict_proba(features_scaled)[0][1] * 100
            st.session_state.audio_score = prob

            label = "😣 Stressed Voice" if prob > 50 else "😌 Calm Voice"
            color = "red" if prob > 50 else "green"

            # Waveform visualization
            y, sr = librosa.load("temp.wav", sr=None)
            fig, ax = plt.subplots(figsize=(8, 2))
            librosa.display.waveshow(y, sr=sr, ax=ax, color="cyan")
            ax.set_title("Waveform of Uploaded Audio")
            ax.set_xlabel("Time (s)")
            st.pyplot(fig)

            st.markdown(f"""
            <div style="background-color:{color};padding:15px;border-radius:10px;">
                <h3 style="color:white;">{label}</h3>
                <p style="color:white;">Stress Probability: <b>{prob:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Acoustic Stress Level", f"{prob:.2f}%")
            st.metric("Prediction", label)
            st.success(f"🎧 Acoustic stress score saved: {prob:.2f}%")

# ------------------ COMBINED MULTIMODAL ANALYSIS ------------------
st.markdown("---")
st.subheader("🧩 Multimodal Agentic Decision Fusion")

if st.button("🧠 Run Combined EEG + Audio Analysis"):
    if st.session_state.eeg_score is None or st.session_state.audio_score is None:
        st.warning("Please analyze both EEG and Audio first before fusion.")
    else:
        eeg_score = st.session_state.eeg_score
        audio_score = st.session_state.audio_score
        final_score = 0.6 * eeg_score + 0.4 * audio_score

        label = "🚨 High Stress" if final_score > 50 else "✅ Relaxed"
        color = "red" if final_score > 50 else "green"

        st.markdown(f"""
        <div style="background-color:{color};padding:15px;border-radius:10px;">
            <h3 style="color:white;">{label}</h3>
            <p style="color:white;">Final Combined Stress Index: <b>{final_score:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.metric("EEG Stress", f"{eeg_score:.2f}%")
        st.metric("Acoustic Stress", f"{audio_score:.2f}%")
        st.metric("Final Stress Score", f"{final_score:.2f}%")

        # -------- Agentic Recommendations --------
        st.markdown("### 🤖 Agentic AI Recommendations")
        agent = recommend_activities(final_score)
        st.markdown(f"**{agent['status']}**")
        for rec in agent["recommendations"]:
            st.write(f"- {rec}")

        # Optional: play relaxation music for high stress
        if final_score > 70:
            if os.path.exists("relax_sounds/relax1.mp3"):
                st.audio("relax_sounds/relax1.mp3", format="audio/mp3")

        # -------- Log & Visualization --------
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "EEG": eeg_score,
            "Audio": audio_score,
            "Fusion": final_score
        })

        df_hist = pd.DataFrame(st.session_state.history)
        if len(df_hist) > 1:
            st.markdown("### 📈 Stress Trend Over Time")
            st.line_chart(df_hist.set_index("timestamp"))

st.markdown("---")
st.caption("Developed by Jairam | Powered by DSP + AI + Streamlit 🧠")
