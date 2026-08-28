import numpy as np
import librosa

def mix_at_snr(clean, noise, snr_db):
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt((clean_power / (10 ** (snr_db / 10))) / noise_power)
    return clean + scale * noise

def generate_pair(clean_path, noise_path, snr_db, sr=16000):
    clean, _ = librosa.load(clean_path, sr=sr)
    noise, _ = librosa.load(noise_path, sr=sr)
    return clean, mix_at_snr(clean, noise, snr_db)
