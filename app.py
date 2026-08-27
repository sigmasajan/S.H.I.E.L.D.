import streamlit as st
import numpy as np
import io
import soundfile as sf

# -------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------
SAMPLE_RATE = 48000  # DeepFilterNet2 native processing rate
DURATION = 4.0       # Standard model chunk length in seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# -------------------------------------------------------------
# STREAMLIT PAGE SETUP
# -------------------------------------------------------------
st.set_page_config(
    page_title="DeepFilterNet2 Audio Evaluation Lab",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ DeepFilterNet2 Audio Evaluation Lab")
st.markdown(
    "Synthesize complex acoustic environments using **LibriSpeech**, **UrbanSound8K**, "
    "and **DEMAND**, then run deep filtering metrics in real-time."
)

# -------------------------------------------------------------
# HELPER FUNCTIONS (MOCK DATA GENERATION)
# -------------------------------------------------------------
def generate_mock_signal(signal_type, frequency=440.0):
    """Generates synthetic audio signals representing different dataset components."""
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    
    if signal_type == "speech":
        # Simulating speech variations using an amplitude-modulated sine wave
        mod = 0.5 * (1.0 + np.sin(2 * np.pi * 1.5 * t))
        signal = np.sin(2 * np.pi * frequency * t) * mod
    elif signal_type == "transient":
        # Simulating an UrbanSound8K burst (e.g., dog bark or car horn)
        signal = np.zeros_like(t)
        start, end = int(NUM_SAMPLES * 0.3), int(NUM_SAMPLES * 0.5)
        signal[start:end] = np.sin(2 * np.pi * 880.0 * t[start:end]) * 0.6
    elif signal_type == "ambient":
        # Simulating a DEMAND stationary room texture (pink/white noise)
        signal = np.random.normal(0, 0.15, NUM_SAMPLES)
    else:
        signal = np.zeros_like(t)
        
    return signal.astype(np.float32)

def to_audio_bytes(audio_array, sr=SAMPLE_RATE):
    """Converts a NumPy floating point array into playable WAV bytes."""
    virtual_file = io.BytesIO()
    sf.write(virtual_file, audio_array, sr, format='WAV', subtype='PCM_16')
    return virtual_file.getvalue()

# -------------------------------------------------------------
# SIDEBAR CONTROLS (DATASET & MODEL CONFIGURATION)
# -------------------------------------------------------------
st.sidebar.header("🎛️ Pipeline Settings")

st.sidebar.subheader("Dataset Mix Ratios")
speech_file = st.sidebar.selectbox(
    "LibriSpeech Target Voice", 
    ["speaker_1082_clean.wav", "speaker_4502_clean.wav", "Upload Custom Audio..."]
)

urban_noise = st.sidebar.selectbox(
    "UrbanSound8K (Transient)", 
    ["01_dog_barking.wav", "02_car_horn.wav", "03_jackhammer.wav", "None"]
)

demand_noise = st.sidebar.selectbox(
    "DEMAND (Ambient Environment)", 
    ["DKITCHEN_16k.wav", "OOFFICE_16k.wav", "PSTATION_16k.wav", "None"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Acoustic Mixing Parameters")
target_snr = st.sidebar.slider("Global Target SNR (dB)", min_value=-10, max_value=25, value=5, step=1)
transient_gain = st.sidebar.slider("UrbanSound8K Scale Factor (α)", 0.0, 1.5, 0.7, 0.1)
ambient_gain = st.sidebar.slider("DEMAND Scale Factor (β)", 0.0, 1.5, 0.4, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("DeepFilterNet2 Engine")
post_filter_attenuation = st.sidebar.slider("Max Attenuation (dB)", 0, 40, 20, 5)
enable_erb = st.sidebar.checkbox("Stage 1: Enforce ERB Envelope", value=True)
enable_df = st.sidebar.checkbox("Stage 2: Enforce Deep Filtering", value=True)

# -------------------------------------------------------------
# MAIN APP BODY: PROCESSING PIPELINE
# -------------------------------------------------------------

# 1. Synthesize the incoming source signals
clean_speech = generate_mock_signal("speech", frequency=350.0)
transient_src = generate_mock_signal("transient") if urban_noise != "None" else np.zeros(NUM_SAMPLES)
ambient_src = generate_mock_signal("ambient") if demand_noise != "None" else np.zeros(NUM_SAMPLES)

# 2. Compute the mixing metrics (Simulating target SNR adjustments)
# Mathematically scale relative to speech power
speech_power = np.mean(clean_speech ** 2) + 1e-8
noise_raw = (transient_src * transient_gain) + (ambient_src * ambient_gain)
noise_raw_power = np.mean(noise_raw ** 2) + 1e-8

snr_factor = 10 ** (target_snr / 10.0)
scale = np.sqrt(speech_power / (noise_raw_power * snr_factor))
scaled_noise = noise_raw * scale

# Construct final corrupted composite wave
noisy_mixture = clean_speech + scaled_noise
# Peak check normalization to prevent clipping artifacts
max_val = np.max(np.abs(noisy_mixture))
if max_val > 0.95:
    noisy_mixture = (noisy_mixture / max_val) * 0.95

# 3. Simulate DeepFilterNet2 inference filtering block
if not enable_erb and not enable_df:
    enhanced_output = noisy_mixture.copy()  # No filtering done
elif enable_erb and not enable_df:
    enhanced_output = clean_speech + (scaled_noise * 0.4)  # Rough background reduction
else:
    # Full Stage 1 + Stage 2 processing reconstruction simulation
    enhanced_output = clean_speech + (scaled_noise * 0.08)

# -------------------------------------------------------------
# INTERACTIVE WORKSPACE RENDER
# -------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.header("📥 Raw Environment Input")
    st.markdown("**Combined Noisy Mixture (LibriSpeech + Noise Sources)**")
    st.audio(to_audio_bytes(noisy_mixture), format="audio/wav")
    
    with st.expander("Examine Source Mix Details"):
        st.caption("Clean Target Speech component:")
        st.audio(to_audio_bytes(clean_speech), format="audio/wav")
        st.caption("Combined Noise component (Urban + DEMAND scaled):")
        st.audio(to_audio_bytes(scaled_noise), format="audio/wav")

with col2:
    st.header("📤 DeepFilterNet2 Output")
    st.markdown("**Processed Enhanced Speech Result**")
    st.audio(to_audio_bytes(enhanced_output), format="audio/wav")
    
    # Render interactive diagnostic model metrics
    st.subheader("📊 Engine Metrics (Simulation)")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    # Simulating dynamic improvements based on architecture settings
    pesq_gain = 1.4 if (enable_df and enable_erb) else (0.5 if enable_erb else 0.0)
    rtf_val = 0.041 if enable_df else 0.012
    
    m_col1.metric(label="Simulated PESQ Score", value=f"{2.1 + pesq_gain:.2f}", delta=f"+{pesq_gain:.2f}" if pesq_gain > 0 else None)
    m_col2.metric(label="Real-Time Factor (RTF)", value=f"{rtf_val} RTF", delta="CPU Safe", delta_color="normal")
    m_col3.metric(label="Processing Latency", value="20.0 ms", delta="Fixed Lookahead")

st.markdown("---")
st.info(
    "💡 **Next Steps for integration:** Replace the `generate_mock_signal` functions with genuine data loading "
    "from your local directories, and drop your `df.enhance` model call inside the processing section."
)
