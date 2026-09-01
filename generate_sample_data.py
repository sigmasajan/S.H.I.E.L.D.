"""
Generates a small SAMPLE dataset entirely locally — no downloads, no
LibriSpeech, no UrbanSound8K.

Run once:  python scripts/generate_sample_data.py
"""

import os
import subprocess
import numpy as np
import soundfile as sf

SR = 16000
CLEAN_DIR = "data/clean"
NOISE_DIR = "data/noise"
MIXED_DIR = "data/mixed"
for d in (CLEAN_DIR, NOISE_DIR, MIXED_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Clean "speech" — real synthesized speech via espeak-ng, short tactical-style
# commands. Robotic-sounding, but genuinely speech (real formants, real words),
# which matters far more than a sine wave for PESQ/STOI to mean anything.
# ---------------------------------------------------------------------------
PHRASES = [
    "hold position and report status",
    "moving to cover, wait for signal",
    "target confirmed, requesting backup",
    "all clear, proceeding north",
]

def make_clean_speech():
    paths = []
    for i, phrase in enumerate(PHRASES):
        path = f"{CLEAN_DIR}/speech_{i:02d}.wav"
        try:
            subprocess.run(
                ["espeak-ng", "-s", "150", "-v", "en", phrase, "-w", path],
                check=True, capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            # espeak-ng not installed on this machine (Linux: sudo apt install
            # espeak-ng, Mac: brew install espeak). Fall back to a synthetic
            # voice-like tone so the script still produces something usable
            # rather than crashing.
            print(f"  espeak-ng not available, using synthetic fallback for '{phrase}'")
            dur = 1.5 + 0.1 * len(phrase.split())
            t = np.linspace(0, dur, int(SR * dur))
            envelope = np.abs(np.sin(2 * np.pi * 2.5 * t))
            tone = 0.3 * np.sin(2 * np.pi * 160 * t) * envelope
            sf.write(path, tone.astype(np.float32), SR)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Noise — synthesized to be clearly one regime or the other, so the demo
# visibly proves the regime detector is doing its job.
# ---------------------------------------------------------------------------
def make_engine_hum(duration=4.0, sr=SR):
    """Steady, low-frequency, periodic — should classify as steady-hum."""
    t = np.linspace(0, duration, int(sr * duration))
    hum = (
        0.5 * np.sin(2 * np.pi * 90 * t)
        + 0.3 * np.sin(2 * np.pi * 180 * t)
        + 0.1 * np.random.randn(len(t))
    )
    return (hum / np.max(np.abs(hum))).astype(np.float32)


def make_gunfire_burst(duration=4.0, sr=SR):
    """Mostly silence with a few sharp, high-amplitude broadband spikes —
    should classify as impulsive."""
    n = int(sr * duration)
    audio = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(5):
        start = rng.integers(0, n - 400)
        burst = rng.standard_normal(300) * np.hanning(300)
        audio[start:start + 300] += (burst * 3.0).astype(np.float32)
    return np.clip(audio, -1, 1)


def make_rotor_wash(duration=4.0, sr=SR):
    """Periodic blade-passing modulation over broadband noise — a 'mixed'
    case, between steady and impulsive."""
    t = np.linspace(0, duration, int(sr * duration))
    broadband = np.random.randn(len(t)) * 0.3
    blade_mod = 0.5 * (1 + np.sin(2 * np.pi * 12 * t))  # ~12 Hz blade passing
    rotor = broadband * blade_mod
    return (rotor / np.max(np.abs(rotor))).astype(np.float32)


def make_noise_files():
    files = {
        f"{NOISE_DIR}/engine_hum.wav": make_engine_hum(),
        f"{NOISE_DIR}/gunfire_burst.wav": make_gunfire_burst(),
        f"{NOISE_DIR}/rotor_wash.wav": make_rotor_wash(),
    }
    for path, audio in files.items():
        sf.write(path, audio, SR)
    return list(files.keys())


# ---------------------------------------------------------------------------
# Mixed pairs — uses the real, already-tested mix_generator.py, so this
# exercises the actual project code, not a separate one-off mixing routine.
# ---------------------------------------------------------------------------
def make_mixed_pairs(clean_paths, noise_paths):
    import sys
    sys.path.insert(0, ".")
    from src.mix_generator import generate_pair

    for clean_path in clean_paths:
        for noise_path in noise_paths:
            snr = 3  # deliberately hard — proves the pipeline under real stress
            clean_name = os.path.splitext(os.path.basename(clean_path))[0]
            noise_name = os.path.splitext(os.path.basename(noise_path))[0]
            _, mixed = generate_pair(clean_path, noise_path, snr_db=snr, sr=SR)
            out_path = f"{MIXED_DIR}/{clean_name}__{noise_name}__{snr}dB.wav"
            sf.write(out_path, mixed, SR)
            print("wrote", out_path)


if __name__ == "__main__":
    print("Generating sample clean speech (espeak-ng)...")
    clean_paths = make_clean_speech()

    print("Generating sample noise (synthesized)...")
    noise_paths = make_noise_files()

    print("Generating mixed pairs (using the real mix_generator.py)...")
    make_mixed_pairs(clean_paths, noise_paths)

    print("\nDone. This is placeholder data to unblock development — swap in")
    print("real LibriSpeech dev-clean + real noise recordings when you can.")
