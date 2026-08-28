"""
S.H.I.E.L.D. — Mix Generator
Generates noisy audio presets by mixing clean speech with noise at a target SNR.
Outputs go to data/mixed/ where app.py loads its bundled scenario presets.

Usage:
  python -m src.mix_generator                        # generate all 4 presets (synthetic)
  python -m src.mix_generator --snr -10              # custom SNR
  python -m src.mix_generator --speech clean.wav     # use a real speech file
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCENARIOS = ["Vehicle_Engine", "Gunfire_Burst", "Helicopter_Rotor", "Mixed_-_Unknown"]
DEFAULT_SR = 16_000
DEFAULT_SNR_DB = -5.0
OUTPUT_DIR = Path("data/mixed")


# ---------------------------------------------------------------------------
# Core mixing
# ---------------------------------------------------------------------------
def mix_at_snr(speech, noise, snr_db):
    """Mix speech and noise arrays at a target SNR (dB)."""
    # Match lengths
    target_len = max(len(speech), len(noise))
    if len(speech) < target_len:
        speech = np.tile(speech, int(np.ceil(target_len / len(speech))))[:target_len]
    if len(noise) < target_len:
        noise = np.tile(noise, int(np.ceil(target_len / len(noise))))[:target_len]
    else:
        noise = noise[:target_len]
    speech = speech[:target_len]

    # Scale noise to achieve target SNR
    speech_rms = np.sqrt(np.mean(speech ** 2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-12)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    mix = speech + noise * (target_noise_rms / noise_rms)

    # Peak-normalise to [-1, 1]
    peak = np.max(np.abs(mix))
    if peak > 1e-12:
        mix /= peak
    return mix


def generate_pair(clean_path, noise_path, snr_db, sr=16000):
    """Load clean speech and noise from disk and return (clean, mixed)."""
    clean, _ = librosa.load(str(clean_path), sr=sr, mono=True)
    noise, _ = librosa.load(str(noise_path), sr=sr, mono=True)
    return clean, mix_at_snr(clean, noise, snr_db)


# ---------------------------------------------------------------------------
# Synthetic noise generators (one per scenario)
# ---------------------------------------------------------------------------
def _synth_noise(scenario, n_samples, sr):
    """Generate simple synthetic noise for a given scenario."""
    t = np.arange(n_samples) / sr

    if scenario == "Vehicle_Engine":
        # Low-frequency rumble
        noise = np.sin(2 * np.pi * 30 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
        noise += 0.3 * np.random.randn(n_samples)

    elif scenario == "Gunfire_Burst":
        # Random impulsive spikes
        noise = np.random.randn(n_samples) * 0.05
        for _ in range(8):
            pos = np.random.randint(0, n_samples)
            burst_len = int(0.01 * sr)
            end = min(pos + burst_len, n_samples)
            noise[pos:end] += np.random.uniform(0.8, 1.0) * np.exp(-np.linspace(0, 5, end - pos))

    elif scenario == "Helicopter_Rotor":
        # Blade-pass thump + whine
        blade_hz = 17.0
        noise = 0.6 * np.sin(2 * np.pi * blade_hz * t)
        noise += 0.3 * np.sin(2 * np.pi * 1200 * t)
        noise += 0.2 * np.random.randn(n_samples)

    else:  # Mixed_-_Unknown
        noise = np.random.randn(n_samples) * 0.5
        noise += 0.3 * np.sin(2 * np.pi * 50 * t)

    return noise / (np.max(np.abs(noise)) + 1e-12)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S.H.I.E.L.D. mix generator")
    parser.add_argument("--speech", type=Path, default=None, help="Clean speech .wav file")
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR_DB, help="Target SNR in dB")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="Sample rate")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration (s) for synthetic speech")
    parser.add_argument("-o", "--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    # Load or synthesise speech
    if args.speech and args.speech.exists():
        speech, _ = librosa.load(str(args.speech), sr=args.sr, mono=True)
        print(f"Loaded speech: {args.speech}  ({len(speech)/args.sr:.1f}s)")
    else:
        n = int(args.duration * args.sr)
        t = np.arange(n) / args.sr
        speech = np.sin(2 * np.pi * 200 * t) * (1 + np.sin(2 * np.pi * 2 * t)) * 0.5
        print(f"Using synthetic speech ({args.duration:.1f}s)")

    # Generate all scenario presets
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(speech)

    for scenario in SCENARIOS:
        noise = _synth_noise(scenario, n_samples, args.sr)
        mix = mix_at_snr(speech, noise, args.snr)
        out_path = args.output_dir / f"{scenario}.wav"
        sf.write(str(out_path), mix, args.sr)
        print(f"✓ {out_path}  ({len(mix)/args.sr:.1f}s, SNR={args.snr:+.1f} dB)")

    print(f"\nDone — {len(SCENARIOS)} presets written to {args.output_dir}/")
