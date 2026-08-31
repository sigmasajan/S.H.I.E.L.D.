"""
S.H.I.E.L.D. — Adaptive Voice-Preserving ANC
Streamlit dashboard: live simulation of the final prototype (main playbook §5).
Owner: Software Engineering. Uses src/enhance.py, src/nlms_filter.py,
src/regime_detector.py, src/metrics.py — see main playbook §5.1.4 for those.
"""

import numpy as np
import librosa
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

from src.enhance import clean_file
from src.nlms_filter import NLMSFilter
from src.regime_detector import classify_regime
from src.metrics import compute_metrics

st.set_page_config(page_title="S.H.I.E.L.D.", layout="wide")

# ---------------------------------------------------------------------------
# Basic auth gate — Cybersecurity owns this (see individual action plans, Day 2).
# TODO(Cyber): swap DASHBOARD_PASSWORD for st.secrets["password"] before this
# goes anywhere public. Fine as a hardcoded string for now.
# ---------------------------------------------------------------------------
DASHBOARD_PASSWORD = "shield2026"

def check_password():
    if st.session_state.get("authed"):
        return True
    pw = st.text_input("Dashboard password", type="password")
    if pw == DASHBOARD_PASSWORD:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Wrong password")
    return False

if not check_password():
    st.stop()

# ---------------------------------------------------------------------------
# Scenario presets — the "tactile mission-mode dial" (differentiator #6),
# simulated as a dropdown. Each preset biases the regime detector exactly like
# the physical rotary dial would (main playbook §5.1.6).
# ---------------------------------------------------------------------------
SCENARIOS = {
    # flatness_thresh values recalibrated against scripts/generate_sample_data.py
    # output — see src/regime_detector.py comment. Re-check once real recordings
    # replace the sample dataset.
    "Vehicle Engine":   {"impulsive_thresh": 7.0, "flatness_thresh": 0.70},
    "Gunfire Burst":    {"impulsive_thresh": 4.5, "flatness_thresh": 0.65},
    "Helicopter Rotor": {"impulsive_thresh": 6.5, "flatness_thresh": 0.65},
    "Mixed / Unknown":  {"impulsive_thresh": 6.0, "flatness_thresh": 0.65},
}

st.title("S.H.I.E.L.D. — Adaptive Voice-Preserving ANC")
st.caption("Live simulation of the final prototype · SIH26052")

with st.sidebar:
    st.header("Controls")
    scenario = st.selectbox("Mission mode (simulated dial)", list(SCENARIOS.keys()))
    bypass = st.checkbox("Bypass — simulate power/compute loss", value=False)
    st.divider()
    st.subheader("Input source")
    input_mode = st.radio("Choose input", ["Upload a clip", "Record live (mic)", "Use a bundled preset"])

# ---------------------------------------------------------------------------
# Get the noisy input as a numpy array + sample rate
# ---------------------------------------------------------------------------
noisy_audio, sr = None, None

if input_mode == "Upload a clip":
    uploaded = st.file_uploader("Noisy .wav clip", type=["wav"])
    if uploaded:
        noisy_audio, sr = librosa.load(uploaded, sr=None, mono=True)

elif input_mode == "Record live (mic)":
    recording = st.audio_input("Speak now, with noise playing in the room")
    if recording:
        noisy_audio, sr = librosa.load(recording, sr=None, mono=True)

else:
    import glob
    preset_files = sorted(glob.glob("data/mixed/*.wav"))
    if preset_files:
        chosen_preset = st.selectbox("Preset file", preset_files)
        noisy_audio, sr = librosa.load(chosen_preset, sr=None, mono=True)
    else:
        st.warning("No .wav files in data/mixed/ yet — ask CS Core 1, or upload/record your own clip instead.")

# Optional simulated second channel — differentiator #2 (dual-channel fusion),
# demoed per §5.1.6 since we don't have a real throat mic yet. App still runs
# fine without it (AI-only path), it just skips the classical-filter blend.
st.sidebar.divider()
ref_upload = st.sidebar.file_uploader("Optional: reference/throat-mic channel (.wav)", type=["wav"], key="ref")
clean_reference = st.sidebar.file_uploader("Optional: ground-truth clean speech, for scoring", type=["wav"], key="clean")

run = st.button("Run S.H.I.E.L.D.", type="primary", disabled=(noisy_audio is None))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def match_length(audio, target_len):
    if len(audio) >= target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))


def build_mode_trace(audio, sr, thresholds, frame_ms=20):
    frame_len = int(sr * frame_ms / 1000)
    return [
        classify_regime(audio[start:start + frame_len], **thresholds)
        for start in range(0, len(audio) - frame_len, frame_len)
    ]


def regime_aware_blend(noisy, ai_cleaned, ref, sr, thresholds, frame_ms=20):
    """Where a frame is classified steady-hum, use the fast NLMS path;
    everywhere else, use the AI path. This is the regime-aware architecture
    from main playbook §2.1/§2.3, computed frame-by-frame in one pass since
    this is a simulation, not the streaming final firmware."""
    frame_len = int(sr * frame_ms / 1000)
    nlms = NLMSFilter()
    out = np.copy(ai_cleaned)
    for start in range(0, len(noisy) - frame_len, frame_len):
        frame = noisy[start:start + frame_len]
        if classify_regime(frame, **thresholds) == "steady-hum":
            ref_frame = ref[start:start + frame_len] if ref is not None else frame
            out[start:start + frame_len] = nlms.process(ref_frame, frame)
    return out


def estimate_relative_exposure_db(audio, sr, frame_ms=100):
    """Rolling relative loudness in dBFS (relative to digital full-scale — NOT
    calibrated dB SPL; a laptop/phone mic isn't a sound-level meter). This is a
    stand-in for ECE 2's exposure telemetry (individual action plan, Day 2) —
    good enough to demo the concept, not a real safety measurement yet."""
    frame_len = int(sr * frame_ms / 1000)
    levels = []
    for start in range(0, len(audio) - frame_len, frame_len):
        rms = np.sqrt(np.mean(audio[start:start + frame_len] ** 2) + 1e-12)
        levels.append(20 * np.log10(rms + 1e-12))
    return levels


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------
if run and noisy_audio is not None:
    mode_trace = None

    if bypass:
        st.warning(
            "BYPASS ACTIVE — playing raw, unprocessed audio. This simulates the "
            "deterministic hardware failsafe (differentiator #3): zero software "
            "in the loop, mic routed straight to output."
        )
        cleaned_audio = noisy_audio

    else:
        with st.spinner("Running the pipeline — regime detection, then AI/classical routing..."):
            # 1. AI path — always computed, it's the safety net for the hard cases
            sf.write("_tmp_noisy.wav", noisy_audio, sr)
            clean_file("_tmp_noisy.wav", "_tmp_cleaned.wav")
            ai_cleaned, _ = librosa.load("_tmp_cleaned.wav", sr=sr)
            ai_cleaned = match_length(ai_cleaned, len(noisy_audio))

            # 2. Regime trace, for the chart
            thresholds = SCENARIOS[scenario]
            mode_trace = build_mode_trace(noisy_audio, sr, thresholds)

            # 3. Blend in the classical filter wherever the reference channel
            #    says "steady-hum" — only possible if a reference was provided
            if ref_upload is not None:
                ref_audio, _ = librosa.load(ref_upload, sr=sr, mono=True)
                ref_audio = match_length(ref_audio, len(noisy_audio))
                cleaned_audio = regime_aware_blend(noisy_audio, ai_cleaned, ref_audio, sr, thresholds)
                st.caption("Regime-aware blend active: NLMS on steady-hum frames, DeepFilterNet2 elsewhere.")
            else:
                cleaned_audio = ai_cleaned
                st.caption(
                    "No reference/throat-mic channel provided — showing AI-only "
                    "enhancement. Upload one in the sidebar to see the full "
                    "regime-aware blend."
                )

    # -----------------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before")
        sf.write("_tmp_before.wav", noisy_audio, sr)
        st.audio("_tmp_before.wav")
    with col2:
        st.subheader("After")
        sf.write("_tmp_after.wav", cleaned_audio, sr)
        st.audio("_tmp_after.wav")

    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax[0].plot(noisy_audio, linewidth=0.5)
    ax[0].set_title("Before — waveform")
    ax[1].plot(cleaned_audio, linewidth=0.5, color="green")
    ax[1].set_title("After — waveform")
    st.pyplot(fig)

    if mode_trace:
        label_to_num = {"steady-hum": 0, "mixed": 1, "impulsive": 2}
        st.subheader("Regime detector — which path handled each moment")
        st.line_chart([label_to_num[m] for m in mode_trace])
        st.caption("0 = classical NLMS path · 1 = mixed/AI path · 2 = impulsive/AI path")

    st.subheader("Relative loudness over time (exposure telemetry stand-in)")
    st.line_chart(estimate_relative_exposure_db(noisy_audio, sr))
    st.caption("dBFS, relative to digital full scale — not a calibrated SPL reading. ECE 2 owns the real version.")

    if clean_reference is not None and not bypass:
        clean_ref_audio, _ = librosa.load(clean_reference, sr=sr, mono=True)
        clean_ref_audio = match_length(clean_ref_audio, len(cleaned_audio))
        metrics = compute_metrics(clean_ref_audio, cleaned_audio, sr)
        m1, m2, m3 = st.columns(3)
        m1.metric("SNR improvement", f"{metrics['snr']:.1f} dB")
        m2.metric("STOI", f"{metrics['stoi']:.2f}")
        m3.metric("PESQ", f"{metrics['pesq']:.2f}")
    elif not bypass:
        st.info("Upload a ground-truth clean-speech reference in the sidebar to see real PESQ/STOI/SNR numbers.")

elif noisy_audio is None:
    st.info("Choose an input source in the sidebar, then press Run S.H.I.E.L.D..")
