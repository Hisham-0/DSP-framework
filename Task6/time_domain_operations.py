"""
Time Domain Operations - Manual Implementations
All operations implemented without using built-in numpy functions
"""
import numpy as np
import math


class TimeDomainOperations:
    """Manual implementations of all 10 time domain operations"""

    @staticmethod
    def smoothing_moving_average(x, window):
        """
        Manual moving average smoothing
        Returns only N-window+1 samples (matching test files)
        """
        N = len(x)
        if N == 0:
            return np.array([])

        if window <= 0 or window > N:
            raise ValueError("Invalid window size")

        # For test files, output has N-window+1 samples
        L = N - window + 1
        y = np.zeros(L)

        # Manual moving average - only compute where full window fits
        for i in range(L):
            window_sum = 0.0
            for j in range(window):
                window_sum += x[i + j]
            y[i] = window_sum / window

        return y

    @staticmethod
    def first_derivative(x):
        """
        Manual first derivative: y[n] = x[n] - x[n-1]

        Parameters:
        -----------
        x : numpy array
            Input signal samples

        Returns:
        --------
        numpy array
            First derivative
        """
        N = len(x)
        if N <= 1:
            return np.array([])

        y = np.zeros(N)
        y[0] = x[0] - 0  # Or use forward difference at boundaries

        for i in range(1, N):
            y[i] = x[i] - x[i - 1]
        return y

    @staticmethod
    def second_derivative(x):
        """
        Manual second derivative: y[n] = x[n+1] - 2x[n] + x[n-1]

        Parameters:
        -----------
        x : numpy array
            Input signal samples

        Returns:
        --------
        numpy array
            Second derivative
        """
        N = len(x)
        if N < 3:
            return np.array([])

        y = np.zeros(N)
        # Handle boundaries separately
        y[0] = x[1] - 2 * x[0] + 0  # Assuming x[-1] = 0
        for i in range(1, N - 1):
            y[i] = x[i + 1] - 2 * x[i] + x[i - 1]
        y[N - 1] = 0 - 2 * x[N - 1] + x[N - 2]  # Assuming x[N] = 0

        return y

    @staticmethod
    def shift_signal(x, k):
        """
        Manual signal shifting

        Parameters:
        -----------
        x : numpy array
            Input signal samples
        k : int
            Shift amount (positive for delay, negative for advance)

        Returns:
        --------
        numpy array
            Shifted signal
        """
        N = len(x)
        if N == 0:
            return np.array([])

        y = np.zeros(N)

        if k == 0:
            return x.copy()

        if k > 0:  # Delay (right shift)
            for i in range(N):
                if i >= k:
                    y[i] = x[i - k]
                else:
                    y[i] = 0.0
        else:  # Advance (left shift)
            k_abs = abs(k)
            for i in range(N):
                if i < N - k_abs:
                    y[i] = x[i + k_abs]
                else:
                    y[i] = 0.0

        return y

    @staticmethod
    def fold_signal(x):
        """
        Manual signal folding (time reversal)

        Parameters:
        -----------
        x : numpy array
            Input signal samples

        Returns:
        --------
        numpy array
            Folded signal
        """
        N = len(x)
        if N == 0:
            return np.array([])

        y = np.zeros(N)

        # Manual reversal
        for i in range(N):
            y[i] = x[N - 1 - i]

        return y

    @staticmethod
    def remove_dc(x):
        """
        Manual DC component removal

        Parameters:
        -----------
        x : numpy array
            Input signal samples

        Returns:
        --------
        numpy array
            Signal with DC removed
        """
        N = len(x)
        if N == 0:
            return np.array([])

        # Manual mean calculation
        signal_sum = 0.0
        for i in range(N):
            signal_sum += x[i]

        mean_val = signal_sum / N

        # Remove DC
        y = np.zeros(N)
        for i in range(N):
            y[i] = x[i] - mean_val

        return y

    @staticmethod
    def convolve_signals(x, h):
        """
        Manual convolution without np.convolve

        Parameters:
        -----------
        x : numpy array
            First signal
        h : numpy array
            Second signal

        Returns:
        --------
        numpy array
            Convolution result
        """
        N = len(x)
        M = len(h)

        if N == 0 or M == 0:
            return np.array([])

        L = N + M - 1
        y = np.zeros(L)

        # Manual convolution
        for n in range(L):
            sum_val = 0.0
            for k in range(N):
                if 0 <= n - k < M:
                    sum_val += x[k] * h[n - k]
            y[n] = sum_val

        return y

    @staticmethod
    def normalized_cross_correlation(x, y):
        """
        Compute normalized cross-correlation between signals x and y.
        Returns lags (from 0 to L-1 where L = N+M-1) and normalized correlation values.
        """
        N = len(x)
        M = len(y)

        if N != M:
            L = N + M - 1
        else:
            L = M

        # Pad signals to length L
        x_padded = np.zeros(L)
        y_padded = np.zeros(L)

        for i in range(N):
            x_padded[i] = x[i]
        for i in range(M):
            y_padded[i] = y[i]

        # Compute correlation for each lag (0 to L-1)
        corr = np.zeros(L)
        for lag in range(L):
            s = 0.0
            for n in range(L):
                # Circular correlation of padded signals
                s += x_padded[n] * y_padded[(n + lag) % L]
            corr[lag] = s / L

        # Compute energies of original signals (not padded)
        x_energy = 0.0
        for i in range(N):
            x_energy += x[i] * x[i]

        y_energy = 0.0
        for i in range(M):
            y_energy += y[i] * y[i]

        # Normalization factor
        norm_factor = math.sqrt(x_energy * y_energy) / L
        if norm_factor == 0:
            norm_factor = 1.0

        # Normalized correlation
        normalized = np.zeros(L)
        for i in range(L):
            normalized[i] = corr[i] / norm_factor

        lags = np.arange(L)
        return lags, normalized

    @staticmethod
    def normalized_autocorrelation(x):
        """
        Manual normalized autocorrelation

        Parameters:
        -----------
        x : numpy array
            Input signal

        Returns:
        --------
        numpy array
            Normalized autocorrelation
        """
        lags, corr = TimeDomainOperations.normalized_cross_correlation(x, x)
        return lags, corr

    @staticmethod
    def circular_cross_correlation_periodic(x, y):
        """
        Manual circular cross-correlation for periodic signals

        Parameters:
        -----------
        x : numpy array
            First signal
        y : numpy array
            Second signal

        Returns:
        --------
        numpy array
            Circular cross-correlation
        """
        N = len(x)
        M = len(y)

        if N == 0 or M == 0:
            return np.array([])

        # Use the maximum length for periodicity
        L = max(N, M)

        # Pad signals to length L by repeating
        x_padded = np.zeros(L)
        y_padded = np.zeros(L)

        for i in range(L):
            x_padded[i] = x[i % N]
            y_padded[i] = y[i % M]

        # Calculate means
        x_mean = np.mean(x_padded)
        y_mean = np.mean(y_padded)

        # Center signals
        x_centered = x_padded - x_mean
        y_centered = y_padded - y_mean

        # Calculate normalization factors
        x_norm = np.sqrt(np.sum(x_centered ** 2))
        y_norm = np.sqrt(np.sum(y_centered ** 2))

        if x_norm == 0 or y_norm == 0:
            return np.zeros(L)

        # Manual circular correlation
        corr = np.zeros(L)

        for lag in range(L):
            sum_val = 0.0
            for n in range(L):
                sum_val += x_centered[n] * y_centered[(n + lag) % L]
            corr[lag] = sum_val / (x_norm * y_norm)

        return corr

    @staticmethod
    def time_delay_analysis_periodic(x, y, ts):
        """
        Manual time delay analysis for periodic signals

        Parameters:
        -----------
        x : numpy array
            First signal
        y : numpy array
            Second signal
        ts : float
            Sampling period in seconds

        Returns:
        --------
        tuple : (signed_lag, time_delay, correlation)
        """
        corr = TimeDomainOperations.circular_cross_correlation_periodic(x, y)
        L = len(corr)

        if L == 0:
            return 0, 0.0, corr

        # Find peak correlation index
        peak_idx = 0
        peak_val = corr[0]

        for i in range(1, L):
            if corr[i] > peak_val:
                peak_val = corr[i]
                peak_idx = i

        # Convert to signed lag (considering circular nature)
        if peak_idx > L // 2:
            signed_lag = peak_idx - L
        else:
            signed_lag = peak_idx

        time_delay = signed_lag * ts

        return signed_lag, time_delay, corr

    def fir_filter(input_signal, filter_type, fs, delta_s, transition_band, f1=None, f2=None):
        """
        Design and apply FIR filter based on specifications.
        
        Parameters:
        - input_signal: numpy array or list of input samples
        - filter_type: 1=Low-pass, 2=High-pass, 3=Band-pass, 4=Band-stop
        - fs: Sampling frequency (Hz)
        - delta_s: Stop-band attenuation (dB)
        - transition_band: Transition band (Hz)
        - fc: Cut-off frequency (Hz) - for Low-pass and High-pass
        - f1, f2: Cut-off frequencies (Hz) - for Band-pass and Band-stop
        
        Returns:
        - h: Filter coefficients
        - output_signal: Filtered signal
        """
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
            h_d = TimeDomainOperations.compute_ideal_lowpass(N, fc_adjusted)
        elif filter_type == 2:  # High-pass
            fc_adjusted = (f1 - half_tb) / fs
            h_d = TimeDomainOperations.compute_ideal_highpass(N, fc_adjusted)
        elif filter_type == 3:  # Band-pass
            f1_adjusted = (f1 - half_tb) / fs
            f2_adjusted = (f2 + half_tb) / fs
            h_d = TimeDomainOperations.compute_ideal_bandpass(N, f1_adjusted, f2_adjusted)
        else:  # Band-stop
            f1_adjusted = (f1 + half_tb) / fs
            f2_adjusted = (f2 - half_tb) / fs
            h_d = TimeDomainOperations.compute_ideal_bandstop(N, f1_adjusted, f2_adjusted)
        
        # Generate window
        w = TimeDomainOperations.generate_window(N, window_type)
        
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
        """
        Resample signal with decimation (M) and interpolation (L) factors.
        Applies low-pass filter to prevent aliasing.
        
        Parameters:
        - input_signal: numpy array or list of input samples
        - M: Decimation factor (0 for no decimation)
        - L: Interpolation factor (0 for no interpolation)
        - fs: Sampling frequency (Hz)
        - delta_s: Stop-band attenuation (dB)
        - transition_band: Transition band (Hz)
        
        Returns:
        - output_signal: Resampled signal
        """
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
        h_d = TimeDomainOperations.compute_ideal_lowpass(N, fc_normalized)
        w = TimeDomainOperations.generate_window(N, window_type)
        h = h_d * w
        
        # Apply resampling based on M and L
        if M == 0 and L != 0:
            # Up-sample by L
            upsampled = TimeDomainOperations.upsample(input_signal, L)
            # Apply low-pass filter
            output_signal = np.convolve(upsampled, h, mode='same')
            
        elif M != 0 and L == 0:
            # Apply low-pass filter first
            filtered = np.convolve(input_signal, h, mode='same')
            # Down-sample by M
            output_signal = TimeDomainOperations.downsample(filtered, M)
            
        else:  # M != 0 and L != 0
            # Up-sample by L
            upsampled = TimeDomainOperations.upsample(input_signal, L)
            # Apply low-pass filter
            filtered = np.convolve(upsampled, h, mode='same')
            # Down-sample by M
            output_signal = TimeDomainOperations.downsample(filtered, M)
        
        return output_signal


    def upsample(signal, L):
        """Up-sample signal by factor L (insert L-1 zeros between samples)"""
        upsampled = np.zeros(len(signal) * L)
        upsampled[::L] = signal
        return upsampled


    def downsample(signal, M):
        """Down-sample signal by factor M (keep every M-th sample)"""
        return signal[::M]