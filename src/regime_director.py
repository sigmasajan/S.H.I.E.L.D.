import numpy as np

# --- YOUR BASE CODE ---
def crest_factor(frame):
    return np.max(np.abs(frame)) / (np.sqrt(np.mean(frame ** 2)) + 1e-9)

def spectral_flatness(frame):
    spec = np.abs(np.fft.rfft(frame)) + 1e-9
    return np.exp(np.mean(np.log(spec))) / np.mean(spec)

def classify_regime(frame, impulsive_thresh=6.0, flatness_thresh=0.3):
    if crest_factor(frame) > impulsive_thresh:
        return "impulsive"
    elif spectral_flatness(frame) < flatness_thresh:
        return "steady-hum"
    return "mixed"

# --- DAY 2 TESTING & THRESHOLD TUNING ---
if __name__ == "__main__":
    print("Running Regime Detector Tests...\n")
    
    sample_rate = 16000
    t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False) # 100ms audio frame
    
    # Test Case 1: Steady Hum (Engine Drone - Pure Sine Wave)
    steady_frame = np.sin(2 * np.pi * 100 * t)
    
    # Test Case 2: Impulsive Noise (Gunshot - Sudden Spike)
    # Create low background noise, then add a massive amplitude spike in the middle
    impulsive_frame = np.random.normal(0, 0.1, len(t))
    impulsive_frame[len(t)//2] = 10.0  
    
    # Test Case 3: Mixed / Chaotic Noise (White Noise)
    mixed_frame = np.random.normal(0, 1.0, len(t))
    
    frames = {
        "Simulated Engine (Should evaluate to 'steady-hum')": steady_frame,
        "Simulated Gunshot (Should evaluate to 'impulsive')": impulsive_frame,
        "Simulated White Noise (Should evaluate to 'mixed')": mixed_frame
    }
    
    for name, frame in frames.items():
        cf = crest_factor(frame)
        sf = spectral_flatness(frame)
        decision = classify_regime(frame)
        
        print(f"--- {name} ---")
        print(f"Calculated Crest Factor: {cf:.2f}")
        print(f"Calculated Spectral Flatness: {sf:.4f}")
        print(f"Routing Decision: -> {decision.upper()} <-\n")
