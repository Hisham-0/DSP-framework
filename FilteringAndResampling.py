import numpy as np

def fir_filter(input_signal, filter_type, fs, delta_s, transition_band, f1=None, f2=None):
    # Convert input to numpy array
    input_signal = np.array(input_signal)
    
    # Validate parameters
    if filter_type in [1, 2] and f1 is None:
        raise ValueError("fc (cut-off frequency) is required for Low-pass and High-pass filters")
    if filter_type in [3, 4] and (f1 is None or f2 is None):
        raise ValueError("f1 and f2 (cut-off frequencies) are required for Band-pass and Band-stop filters")
    
    # Select window based on stop-band attenuation
    if delta_s <= 21:
        window_type = 'rectangular'
        N = int(np.ceil(0.9 / (transition_band / fs)))
    elif delta_s <= 44:
        window_type = 'hanning'
        N = int(np.ceil(3.1 / (transition_band / fs)))
    elif delta_s <= 53:
        window_type = 'hamming'
        N = int(np.ceil(3.3 / (transition_band / fs)))
    else:
        window_type = 'blackman'
        N = int(np.ceil(5.5 / (transition_band / fs)))
    
    # Make N odd for Type-I filter (symmetric coefficients)
    if N % 2 == 0:
        N += 1
    
    # Adjust cut-off frequencies using half transition band
    half_tb = transition_band / 2
    
    if filter_type == 1:  # Low-pass
        fc_adjusted = (f1 + half_tb) / fs
        h_d = compute_ideal_lowpass(N, fc_adjusted)
    elif filter_type == 2:  # High-pass
        fc_adjusted = (f1 - half_tb) / fs
        h_d = compute_ideal_highpass(N, fc_adjusted)
    elif filter_type == 3:  # Band-pass
        f1_adjusted = (f1 - half_tb) / fs
        f2_adjusted = (f2 + half_tb) / fs
        h_d = compute_ideal_bandpass(N, f1_adjusted, f2_adjusted)
    else:  # Band-stop
        f1_adjusted = (f1 + half_tb) / fs
        f2_adjusted = (f2 - half_tb) / fs
        h_d = compute_ideal_bandstop(N, f1_adjusted, f2_adjusted)
    
    # Generate window
    w = generate_window(N, window_type)
    
    # Compute FIR coefficients
    h = h_d * w
    
    # Convolve input signal with coefficients
    output_signal = np.convolve(input_signal, h, mode='same')
    
    # Save coefficients to file
    np.savetxt('fir_coefficients.txt', h, fmt='%.10f')
    
    return h, output_signal


def compute_ideal_lowpass(N, fc):
    """Compute ideal low-pass filter impulse response"""
    n = np.arange(N)
    n_center = (N - 1) / 2
    h_d = np.zeros(N)
    
    for i in range(N):
        if n[i] == n_center:
            h_d[i] = 2 * fc
        else:
            h_d[i] = np.sin(2 * np.pi * fc * (n[i] - n_center)) / (np.pi * (n[i] - n_center))
    
    return h_d


def compute_ideal_highpass(N, fc):
    """Compute ideal high-pass filter impulse response"""
    n = np.arange(N)
    n_center = (N - 1) / 2
    h_d = np.zeros(N)
    
    for i in range(N):
        if n[i] == n_center:
            h_d[i] = 1 - 2 * fc
        else:
            h_d[i] = -np.sin(2 * np.pi * fc * (n[i] - n_center)) / (np.pi * (n[i] - n_center))
    
    return h_d


def compute_ideal_bandpass(N, f1, f2):
    """Compute ideal band-pass filter impulse response"""
    n = np.arange(N)
    n_center = (N - 1) / 2
    h_d = np.zeros(N)
    
    for i in range(N):
        if n[i] == n_center:
            h_d[i] = 2 * (f2 - f1)
        else:
            h_d[i] = (np.sin(2 * np.pi * f2 * (n[i] - n_center)) / (np.pi * (n[i] - n_center)) -
                    np.sin(2 * np.pi * f1 * (n[i] - n_center)) / (np.pi * (n[i] - n_center)))
    
    return h_d


def compute_ideal_bandstop(N, f1, f2):
    """Compute ideal band-stop filter impulse response"""
    n = np.arange(N)
    n_center = (N - 1) / 2
    h_d = np.zeros(N)
    
    for i in range(N):
        if n[i] == n_center:
            h_d[i] = 1 - 2 * (f2 - f1)
        else:
            h_d[i] = (np.sin(2 * np.pi * f1 * (n[i] - n_center)) / (np.pi * (n[i] - n_center)) -
                    np.sin(2 * np.pi * f2 * (n[i] - n_center)) / (np.pi * (n[i] - n_center)))
    
    return h_d


def generate_window(N, window_type):
    """Generate window function"""
    if window_type == 'rectangular':
        return np.ones(N)
    elif window_type == 'hanning':
        #return lambda n, N: 0.5 - 0.5 * np.cos(2.0  * np.pi * n/(N - 1))
        return np.hanning(N)
    elif window_type == 'hamming':
        #return lambda n, N: 0.54 + 0.46 * np.cos(2.0 * np.pi * n/(N - 1))
        return np.hamming(N)
    elif window_type == 'blackman':
        return np.blackman(N)
    else:
        return np.ones(N)


def resampling(input_signal, M, L, fs, delta_s, transition_band):
    # Convert input to numpy array
    input_signal = np.array(input_signal)
    
    # Check for invalid case
    if M == 0 and L == 0:
        raise ValueError("ERROR: Both M and L cannot be zero!")
    
    # Determine cut-off frequency based on M and L
    if M != 0 and L != 0:
        fc = fs / (2 * max(M, L))
    elif M != 0:
        fc = fs / (2 * M)
    else:  # L != 0
        fc = fs / (2 * L)
    
    # Design low-pass filter
    if delta_s <= 21:
        window_type = 'rectangular'
        N = int(np.ceil(0.9 / (transition_band / fs)))
    elif delta_s <= 44:
        window_type = 'hanning'
        N = int(np.ceil(3.1 / (transition_band / fs)))
    elif delta_s <= 53:
        window_type = 'hamming'
        N = int(np.ceil(3.3 / (transition_band / fs)))
    else:
        window_type = 'blackman'
        N = int(np.ceil(5.5 / (transition_band / fs)))
    
    if N % 2 == 0:
        N += 1
    
    # Compute filter coefficients
    fc_normalized = (fc + transition_band / 2) / fs
    h_d = compute_ideal_lowpass(N, fc_normalized)
    w = generate_window(N, window_type)
    h = h_d * w
    
    # Apply resampling based on M and L
    if M == 0 and L != 0:
        # Up-sample by L
        upsampled = upsample(input_signal, L)
        # Apply low-pass filter
        output_signal = np.convolve(upsampled, h, mode='same')
        
    elif M != 0 and L == 0:
        # Apply low-pass filter first
        filtered = np.convolve(input_signal, h, mode='same')
        # Down-sample by M
        output_signal = downsample(filtered, M)
        
    else:  # M != 0 and L != 0
        # Up-sample by L
        upsampled = upsample(input_signal, L)
        # Apply low-pass filter
        filtered = np.convolve(upsampled, h, mode='same')
        # Down-sample by M
        output_signal = downsample(filtered, M)
    
    return output_signal


def upsample(signal, L):
    """Up-sample signal by factor L (insert L-1 zeros between samples)"""
    upsampled = np.zeros(len(signal) * L)
    upsampled[::L] = signal
    return upsampled


def downsample(signal, M):
    """Down-sample signal by factor M (keep every M-th sample)"""
    return signal[::M]