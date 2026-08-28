import numpy as np
from scipy.signal import resample_poly
from pesq import pesq        # verified signature: pesq(fs, ref, deg, mode='wb', on_error=0)
from pystoi import stoi      # verified signature: stoi(x, y, fs_sig, extended=False)

# --- YOUR BASE CODE ---
def to_16k(signal, orig_sr):
    if orig_sr == 16000:
        return signal
    return resample_poly(signal, 16000, orig_sr)

def compute_metrics(clean, test_signal, orig_sr):
    clean16, test16 = to_16k(clean, orig_sr), to_16k(test_signal, orig_sr)
    n = min(len(clean16), len(test16))
    clean16, test16 = clean16[:n], test16[:n]

    p = pesq(16000, clean16, test16, 'wb')
    s = stoi(clean16, test16, 16000, extended=False)
    noise = clean16 - test16
    snr = 10 * np.log10(np.sum(clean16 ** 2) / (np.sum(noise ** 2) + 1e-12))
    return {"pesq": p, "stoi": s, "snr": snr}

# --- EVALUATION & TESTING BLOCK ---
if __name__ == "__main__":
    print("Running Audio Metrics Test...\n")
    
    sample_rate = 16000
    duration = 2.0  # 2 seconds of audio
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 1. Create a "Clean" Reference Signal (Simulated Voice Tone)
    clean_audio = 0.5 * np.sin(2 * np.pi * 400 * t)
    
    # 2. Create a "Bad" Signal (Heavy Static Noise added)
    heavy_noise = np.random.normal(0, 0.5, len(t))
    bad_audio = clean_audio + heavy_noise
    
    # 3. Create a "Good" Signal (Simulating output after your NLMS filter cleans it)
    light_noise = heavy_noise * 0.1  # Noise reduced by 90%
    good_audio = clean_audio + light_noise
    
    # Run the metrics evaluation
    print("--- Evaluating BAD Audio (Raw Mic Input) ---")
    bad_metrics = compute_metrics(clean_audio, bad_audio, sample_rate)
    # PESQ ranges from -0.5 to 4.5 (Higher is better)
    print(f"PESQ (Speech Quality): {bad_metrics['pesq']:.2f}")
    # STOI ranges from 0.0 to 1.0 (Higher is better)
    print(f"STOI (Intelligibility): {bad_metrics['stoi']:.2f}")
    print(f"SNR (Signal-to-Noise): {bad_metrics['snr']:.2f} dB\n")
    
    print("--- Evaluating GOOD Audio (After NLMS Filter) ---")
    good_metrics = compute_metrics(clean_audio, good_audio, sample_rate)
    print(f"PESQ (Speech Quality): {good_metrics['pesq']:.2f}")
    print(f"STOI (Intelligibility): {good_metrics['stoi']:.2f}")
    print(f"SNR (Signal-to-Noise): {good_metrics['snr']:.2f} dB\n")
