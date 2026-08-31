import numpy as np

def crest_factor(frame):
    return np.max(np.abs(frame)) / (np.sqrt(np.mean(frame ** 2)) + 1e-9)

def spectral_flatness(frame):
    spec = np.abs(np.fft.rfft(frame)) + 1e-9
    return np.exp(np.mean(np.log(spec))) / np.mean(spec)

def classify_regime(frame, impulsive_thresh=6.0, flatness_thresh=0.7):
    # flatness_thresh=0.7 calibrated against actual synthesized steady-hum
    # (measured 0.51-0.62) vs. impulsive/mixed noise (measured 0.80+) —
    # the old default of 0.3 never fired on real steady noise. Re-check this
    # once real recordings replace the sample dataset (scripts/generate_sample_data.py).
    if crest_factor(frame) > impulsive_thresh:
        return "impulsive"
    elif spectral_flatness(frame) < flatness_thresh:
        return "steady-hum"
    return "mixed"
