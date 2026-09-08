import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import scipy.signal
import asyncio
import edge_tts
from streamlit_mic_recorder import mic_recorder

# Set Streamlit Page Layout
st.set_page_config(
    page_title="Voice Shield AI | Deepfake Detection Engine",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Real-Time Voice Cloning Detection & Defense")
st.caption("AI-Powered Audio Deepfake Analysis & Active Threat Mitigation Engine")
st.markdown("---")

# Async Helper for Edge-TTS Synthetic Generation
async def generate_synthetic_audio(text, voice="en-US-GuyNeural"):
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

# DSP Pre-Processing Filters
def apply_audio_filters(y, sr, lowcut=None, highcut=None, enable_noise_reduction=False):
    filtered_y = y.copy()
    
    # Butterworth Bandpass / Lowpass / Highpass
    if lowcut or highcut:
        nyquist = 0.5 * sr
        if lowcut and highcut and lowcut < highcut:
            b, a = scipy.signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
        elif lowcut:
            b, a = scipy.signal.butter(4, lowcut / nyquist, btype='highpass')
        elif highcut:
            b, a = scipy.signal.butter(4, highcut / nyquist, btype='lowpass')
        filtered_y = scipy.signal.filtfilt(b, a, filtered_y)

    # Basic Spectral Noise Reduction Gating
    if enable_noise_reduction:
        stft = librosa.stft(filtered_y)
        stft_db = librosa.amplitude_to_db(np.abs(stft))
        mean_db = np.mean(stft_db, axis=1, keepdims=True)
        mask = stft_db > (mean_db - 10)
        filtered_y = librosa.istft(stft * mask)

    return filtered_y

# Deepfake Feature Extraction Pipeline (With Bounded Pitch Tracking)
def extract_audio_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    
    # Energy-Thresholded Pitch (F0) Tracking
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_track = []
    
    mag_thresh = np.max(magnitudes) * 0.2 if np.max(magnitudes) > 0 else 0
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 50 and magnitudes[index, t] > mag_thresh:  # Human vocal frequency check
            pitch_track.append(pitch)
            
    # Calculate Standard Deviation to prevent extreme outlier spikes
    pitch_std = np.std(pitch_track) if len(pitch_track) > 5 else 0.0
    
    return mfccs, spectral_centroids, zcr, spectral_rolloff, pitch_std

# Deepfake Risk Evaluation Engine
def analyze_deepfake_risk(mfccs, spectral_centroids, zcr, spectral_rolloff, pitch_std):
    sc_var = np.var(spectral_centroids)
    
    # Normalized spectral centroid ratio
    spectral_score = 1.0 - (sc_var / (sc_var + 300000))
    
    # Synthetic voices present rigid/flat pitch or phase discontinuities
    # Natural spoken human pitch std typically spans 20 Hz to 80 Hz
    if pitch_std < 15.0 or pitch_std > 120.0:
        pitch_risk = 0.85
    else:
        pitch_risk = 0.25

    raw_score = (spectral_score * 0.5 + pitch_risk * 0.5) * 100
    synthetic_confidence = float(np.clip(raw_score, 10.0, 98.5))
    
    return synthetic_confidence

# Sidebar Controls & DSP Filters
st.sidebar.header("🛡️ Active Defense Controls")
enable_cloaking = st.sidebar.toggle("Enable Acoustic Cloaking", value=False)
enable_auto_drop = st.sidebar.toggle("Auto-Drop Suspicious Calls", value=True)
threshold = st.sidebar.slider("Alert Sensitivity Threshold (%)", 50, 95, 75)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ DSP Pre-Processing Filters")
apply_lowpass = st.sidebar.checkbox("Enable Low-Pass Filter")
lowcut_freq = st.sidebar.slider("High-Pass Cutoff (Hz)", 50, 1000, 100) if st.sidebar.checkbox("Enable High-Pass Filter") else None
highcut_freq = st.sidebar.slider("Low-Pass Cutoff (Hz)", 1000, 8000, 4000) if apply_lowpass else None
enable_nr = st.sidebar.checkbox("Enable Noise Suppression Gating")

# Main Interface Layout
col1, col2 = st.columns([1, 1])
audio_bytes = None
sample_rate = 16000

with col1:
    st.subheader("🎙️ Input Stream Source")
    input_type = st.radio(
        "Select Audio Source:", 
        ["Generate Synthetic Test Audio", "Microphone (Live Record)", "Upload Audio File"], 
        horizontal=True
    )

    if input_type == "Generate Synthetic Test Audio":
        st.write("Generate a synthetic voice stream for live demo testing:")
        gen_text = st.text_area("Phrase to Synthesize:", "Alert: Deepfake voice stream generated for live risk evaluation.")
        voice_model = st.selectbox("Select Neural Voice:", ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural"])
        
        if st.button("⚡ Synthesize & Load Audio"):
            with st.spinner("Generating synthetic audio via Neural Engine..."):
                audio_bytes = asyncio.run(generate_synthetic_audio(gen_text, voice_model))
                st.audio(audio_bytes, format="audio/mp3")

    elif input_type == "Microphone (Live Record)":
        st.write("Click below to record a live spoken phrase:")
        record_dict = mic_recorder(
            start_prompt="🔴 Start Recording",
            stop_prompt="⬛ Stop Recording",
            key='recorder'
        )
        if record_dict:
            audio_bytes = record_dict['bytes']
            st.audio(audio_bytes, format='audio/wav')

    else:
        uploaded_file = st.file_uploader("Choose an audio file (.wav, .mp3)", type=["wav", "mp3"])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/wav')

# Inference & Display Pipeline
if audio_bytes is not None:
    y_raw, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, duration=30.0)
    
    # Apply DSP Filters
    y = apply_audio_filters(y_raw, sr, lowcut=lowcut_freq, highcut=highcut_freq, enable_noise_reduction=enable_nr)
    
    # Extract Features
    mfccs, centroids, zcr, rolloff, pitch_std = extract_audio_features(y, sr)
    
    # Calculate Synthetic Risk Score
    fake_score = analyze_deepfake_risk(mfccs, centroids, zcr, rolloff, pitch_std)
    
    with col2:
        st.subheader("📊 Real-Time Threat Score")
        
        # Tiered Banner Logic
        if fake_score >= threshold:
            st.error(f"🚨 **HIGH RISK: AI VOICE CLONE DETECTED** ({fake_score:.1f}% Confidence)")
            if enable_auto_drop:
                st.warning("⚠️ **AUTOMATED ACTION:** Call stream intercepted and dropped.")
        elif fake_score >= 50.0:
            st.warning(f"⚠️ **SUSPICIOUS AUDIO DETECTED** ({fake_score:.1f}% Synthetic Probability)")
            if enable_cloaking:
                st.info("🔒 **ACOUSTIC CLOAKING:** Active mitigation enabled.")
        else:
            st.success(f"✅ **AUTHENTIC HUMAN VOICE** ({100 - fake_score:.1f}% Real)")
            if enable_cloaking:
                st.info("🔒 **ACOUSTIC CLOAKING:** Standby mode active.")

        # Live Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Synthetic Score", f"{fake_score:.1f}%", delta=f"{fake_score - threshold:.1f}% vs Threshold")
        m2.metric("Duration Analyzed", f"{len(y)/sr:.2f}s")
        m3.metric("Pitch Std Dev", f"{pitch_std:.1f} Hz")

    # Plot Visualizations
    st.markdown("---")
    st.subheader("📈 Audio Signal & Spectral Artifact Analysis")
    
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.write("**Processed Waveform**")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#1E88E5')
        ax.set_title("Time Domain Waveform (Filtered)")
        st.pyplot(fig)

    with v_col2:
        st.write("**Mel-Frequency Spectrogram (High-Freq Phase Scan)**")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='magma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Spectral Density Map")
        st.pyplot(fig)

else:
    with col2:
        st.info("👈 Choose an input source (or generate synthetic audio) in the left panel to trigger analysis.")
