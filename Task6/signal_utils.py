"""
Signal Utility Functions
Helper functions for signal processing
"""
import numpy as np
import math


class SignalUtils:
    """Utility functions for signal processing"""

    @staticmethod
    def parse_signal_string(s):
        """
        Parse string of numbers into numpy array

        Parameters:
        -----------
        s : str
            String containing numbers separated by comma/space/newline

        Returns:
        --------
        numpy array
            Parsed signal
        """
        if s is None or not s.strip():
            return np.array([])

        # Replace various separators with spaces
        s = s.replace(',', ' ').replace(';', ' ').replace('\n', ' ')

        # Split and parse
        parts = s.split()
        numbers = []

        for part in parts:
            try:
                numbers.append(float(part))
            except ValueError:
                continue

        return np.array(numbers)

    @staticmethod
    def load_signal_from_file(file_path):
        """
        Load signal from text file

        Parameters:
        -----------
        file_path : str
            Path to signal file

        Returns:
        --------
        numpy array
            Loaded signal
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return SignalUtils.parse_signal_string(content)
        except Exception as e:
            raise ValueError(f"Failed to load signal from {file_path}: {str(e)}")

    @staticmethod
    def save_signal_to_file(signal, file_path):
        """
        Save signal to text file

        Parameters:
        -----------
        signal : numpy array
            Signal to save
        file_path : str
            Path to save file
        """
        try:
            with open(file_path, 'w') as f:
                for value in signal:
                    f.write(f"{value}\n")
        except Exception as e:
            raise ValueError(f"Failed to save signal to {file_path}: {str(e)}")

    @staticmethod
    def generate_test_signal(signal_type='sine', length=100, amplitude=1.0,
                             frequency=1.0, phase=0.0, noise_level=0.0):
        """
        Generate test signals

        Parameters:
        -----------
        signal_type : str
            Type of signal ('sine', 'cosine', 'square', 'triangle')
        length : int
            Number of samples
        amplitude : float
            Signal amplitude
        frequency : float
            Signal frequency in Hz
        phase : float
            Phase in degrees
        noise_level : float
            Additive noise level

        Returns:
        --------
        tuple : (indices, samples)
        """
        indices = np.arange(length)
        phase_rad = phase * math.pi / 180.0

        if signal_type.lower() == 'sine':
            samples = amplitude * np.sin(2 * math.pi * frequency * indices / length + phase_rad)
        elif signal_type.lower() == 'cosine':
            samples = amplitude * np.cos(2 * math.pi * frequency * indices / length + phase_rad)
        elif signal_type.lower() == 'square':
            samples = amplitude * np.sign(np.sin(2 * math.pi * frequency * indices / length + phase_rad))
        elif signal_type.lower() == 'triangle':
            t = indices / length
            samples = 2 * amplitude * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - amplitude
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, length)
            samples += noise

        return indices, samples

    @staticmethod
    def align_signals(x, y):
        """
        Align two signals to same length

        Parameters:
        -----------
        x : numpy array
            First signal
        y : numpy array
            Second signal

        Returns:
        --------
        tuple : (x_aligned, y_aligned)
        """
        len_x = len(x)
        len_y = len(y)

        if len_x == len_y:
            return x, y

        # Pad shorter signal with zeros
        if len_x < len_y:
            x_aligned = np.zeros(len_y)
            x_aligned[:len_x] = x
            y_aligned = y
        else:
            y_aligned = np.zeros(len_x)
            y_aligned[:len_y] = y
            x_aligned = x

        return x_aligned, y_aligned

    @staticmethod
    def calculate_statistics(signal):
        """
        Calculate basic statistics of signal

        Parameters:
        -----------
        signal : numpy array
            Input signal

        Returns:
        --------
        dict : Statistics dictionary
        """
        if len(signal) == 0:
            return {}

        # Manual calculations
        N = len(signal)

        # Mean
        signal_sum = 0.0
        for i in range(N):
            signal_sum += signal[i]
        mean_val = signal_sum / N

        # Variance
        var_sum = 0.0
        for i in range(N):
            var_sum += (signal[i] - mean_val) ** 2
        variance = var_sum / N

        # Min and max
        min_val = signal[0]
        max_val = signal[0]
        for i in range(1, N):
            if signal[i] < min_val:
                min_val = signal[i]
            if signal[i] > max_val:
                max_val = signal[i]

        return {
            'mean': mean_val,
            'variance': variance,
            'std_dev': math.sqrt(variance),
            'min': min_val,
            'max': max_val,
            'range': max_val - min_val,
            'length': N
        }