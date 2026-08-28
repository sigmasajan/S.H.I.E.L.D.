import numpy as np
import scipy.io.wavfile as wavfile

# --- YOUR BASE CODE (From Screenshot) ---
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

# --- DAY 2 INTEGRATION & TESTING ---
if __name__ == "__main__":
    print("Generating synthetic steady-hum test data...")
    
    # Setup Audio Parameters (16kHz is standard for voice)
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 1. Generate a Steady Hum (Simulating a continuous 100Hz engine drone)
    steady_drone_noise = 0.8 * np.sin(2 * np.pi * 100 * t) 
    
    # 2. Generate a "Voice" (Simulating a 400Hz pure tone for testing)
    clean_voice = 0.5 * np.sin(2 * np.pi * 400 * t)
    
    # 3. Create the Mic Inputs
    # Primary Mic: Picks up the soldier's voice AND the engine drone
    primary_mic = clean_voice + steady_drone_noise
    
    # Reference Mic (Threat Mic): Picks up ONLY the drone
    reference_mic = steady_drone_noise
    
    # 4. Run the NLMS Filter
    print("Processing audio through NLMS Filter...")
    # Using 128 taps for better audio resolution
    nlms = NLMSFilter(num_taps=128, mu=0.1) 
    
    clean_output = nlms.process(reference_mic, primary_mic)
    
    # 5. Save the results as .wav files to listen and verify
    def save_wav(filename, data, rate):
        # Normalize and convert to 16-bit PCM format
        scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        wavfile.write(filename, rate, scaled)
        
    save_wav("test_1_noisy_input.wav", primary_mic, sample_rate)
    save_wav("test_2_clean_output.wav", clean_output, sample_rate)
    
    print("Test Complete! 2 .wav files have been saved in your current folder.")
