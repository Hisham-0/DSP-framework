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
