import numpy as np
from scipy.signal import resample_poly
from pesq import pesq          # verified signature: pesq(fs, ref, deg, mode='wb', on_error=0)
from pystoi import stoi        # verified signature: stoi(x, y, fs_sig, extended=False)

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
