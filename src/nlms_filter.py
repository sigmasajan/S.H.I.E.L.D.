import numpy as np

class NLMSFilter:
    def __init__(self, num_taps=64, mu=0.5, eps=1e-6):
        self.w = np.zeros(num_taps)
        self.mu, self.eps, self.num_taps = mu, eps, num_taps

    def process(self, ref_signal, primary_signal):
        n = len(primary_signal)
        output = np.zeros(n)
        buf = np.zeros(self.num_taps)
        for i in range(n):
            buf[1:] = buf[:-1]
            buf[0] = ref_signal[i]
            e = primary_signal[i] - np.dot(self.w, buf)
            self.w += (self.mu / (np.dot(buf, buf) + self.eps)) * e * buf
            output[i] = e
        return output
