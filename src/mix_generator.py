import numpy as np
import librosa


def mix_at_snr(clean, noise, snr_db):
    """Mix `clean` speech with `noise` at a target SNR (dB). Both are 1-D
    numpy arrays; noise is tiled/trimmed to match the length of clean."""
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]

    clean_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return clean + scale * noise


def generate_pair(clean_path, noise_path, snr_db, sr=16000):
    """Load a clean speech file and a noise file, mix at snr_db, and return
    (clean, mixed) as same-length numpy arrays at sample rate `sr`."""
    clean, _ = librosa.load(clean_path, sr=sr, mono=True)
    noise, _ = librosa.load(noise_path, sr=sr, mono=True)
    mixed = mix_at_snr(clean, noise, snr_db)
    return clean, mixed
