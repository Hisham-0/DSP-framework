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

        Parameters:
        -----------
        x : numpy array
            Input signal samples
        window : int
            Window size (odd number recommended)

        Returns:
        --------
        numpy array
            Smoothed signal
        """
        N = len(x)
        if N == 0:
            return np.array([])

        if window <= 0 or window > N:
            raise ValueError("Invalid window size")

        # Initialize result array
        y = np.zeros(N)
        half_window = window // 2

        # Manual moving average
        for i in range(N):
            # Calculate boundaries for window
            start = max(0, i - half_window)
            end = min(N, i + half_window + 1)

            # Manual sum
            window_sum = 0.0
            count = 0
            for j in range(start, end):
                window_sum += x[j]
                count += 1

            y[i] = window_sum / count if count > 0 else x[i]

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
    def convolve_signals(x, x_indices, h, h_indices):
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
            return np.array([]), np.array([])

        L = N + M - 1
        y = np.zeros(L)

        # Calculate output indices
        start_idx = x_indices[0] + h_indices[0]
        output_indices = np.arange(start_idx, start_idx + L)

        # Manual convolution
        for n in range(L):
            sum_val = 0.0
            for k in range(N):
                if 0 <= n - k < M:
                    sum_val += x[k] * h[n - k]
            y[n] = sum_val

        return output_indices, y  # Return indices too

    @staticmethod
    def normalized_cross_correlation(x, y):
        """
        Manual normalized cross-correlation

        Parameters:
        -----------
        x : numpy array
            First signal
        y : numpy array
            Second signal

        Returns:
        --------
        numpy array
            Normalized cross-correlation
        """
        N = len(x)
        M = len(y)

        if N == 0 or M == 0:
            return np.array([])

        L = N + M - 1
        corr = np.zeros(L)

        # Calculate means manually
        x_sum = 0.0
        y_sum = 0.0

        for i in range(N):
            x_sum += x[i]
        for i in range(M):
            y_sum += y[i]

        x_mean = x_sum / N
        y_mean = y_sum / M

        # Center signals
        x_centered = np.zeros(N)
        y_centered = np.zeros(M)

        for i in range(N):
            x_centered[i] = x[i] - x_mean
        for i in range(M):
            y_centered[i] = y[i] - y_mean

        # Calculate correlation for each lag
        for lag in range(-(M - 1), N):
            idx = lag + (M - 1)  # Map to 0-based index
            sum_val = 0.0

            for n in range(N):
                if 0 <= n + lag < M:
                    sum_val += x_centered[n] * y_centered[n + lag]

            corr[idx] = sum_val / N

        # Calculate normalization factor
        x_sq_sum = 0.0
        y_sq_sum = 0.0

        for i in range(N):
            x_sq_sum += x_centered[i] ** 2
        for i in range(M):
            y_sq_sum += y_centered[i] ** 2

        norm_factor = math.sqrt(x_sq_sum * y_sq_sum)

        if norm_factor != 0:
            corr = corr / norm_factor

        lags = np.arange(-(M - 1), N)

        return lags, corr

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
        return TimeDomainOperations.normalized_cross_correlation(x, x)

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
        lx = len(x)
        ly = len(y)

        if lx == 0 or ly == 0:
            return np.array([])

        # Use the longer length for periodicity
        L = max(lx, ly)

        # Calculate means
        x_mean = sum(x) / lx
        y_mean = sum(y) / ly

        # Center signals
        x_centered = x - x_mean
        y_centered = y - y_mean

        # Manual circular correlation
        corr = np.zeros(L)

        for lag in range(L):
            sum_val = 0.0
            for n in range(L):
                sum_val += x_centered[n % lx] * y_centered[(n + lag) % ly]
            corr[lag] = sum_val / L

        # Normalization
        x_sq_sum = sum(xi ** 2 for xi in x_centered)
        y_sq_sum = sum(yi ** 2 for yi in y_centered)

        norm_factor = math.sqrt(x_sq_sum * y_sq_sum)
        if norm_factor != 0:
            corr = corr / norm_factor

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

        # Manual peak finding
        peak_idx = 0
        peak_val = corr[0]

        for i in range(1, L):
            if corr[i] > peak_val:
                peak_val = corr[i]
                peak_idx = i

        # Convert to signed lag
        lags = np.arange(-L // 2, L // 2 + 1) if L % 2 == 0 else np.arange(-(L - 1) // 2, (L - 1) // 2 + 1)
        peak_idx = np.argmax(corr)
        signed_lag = lags[peak_idx]

        time_delay = signed_lag * ts

        return signed_lag, time_delay, corr
