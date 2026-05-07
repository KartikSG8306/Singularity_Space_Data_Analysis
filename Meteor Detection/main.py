import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def detect_meteors(file_path):
    print("Reading data...")
    df = pd.read_csv(file_path)
    
    
    window_size = 100
    baseline = df['Level'].rolling(window=window_size, center=True).median()
    
    baseline = baseline.bfill().ffill()
    
    signal = df['Level'] - baseline
    
    mad = np.median(np.abs(signal - np.median(signal)))
    robust_std = 1.4826 * mad
    
    threshold = 4.0 * robust_std
    
    peaks, properties = find_peaks(signal, height=threshold, distance=20)
    
    print("-" * 30)
    print("Detection Results:")
    print(f"Total number of meteors detected: {len(peaks)}")
    print(f"Robust Std Dev: {robust_std:.2f} dB")
    print(f"Detection Threshold: {threshold:.2f} dB above baseline")
    print("-" * 30)
    
    return len(peaks)

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data.csv")
    detect_meteors(file_path)
