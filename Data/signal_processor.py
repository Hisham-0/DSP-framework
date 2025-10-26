"""
Signal Processing Framework - Logic Layer
Enhanced with additional operations
"""
import math

import numpy as np


class SignalProcessor:
    """Core signal processing operations"""

    @staticmethod
    def parse_signal_file(file_path):
        """
        that handel the all lines of the file.
        file format:
        Line 1: Signal Type 0/Time or 1/Frequency
        Line 2: Periodicity (0/1)
        Line 3: Num Samples N
        Line 4+: Data (Index Amplitude)
        And return a dictionary after sampling .
        """
        try:
            with open(file_path, 'r') as file:  # 'r' read mode
                lines = [line.strip() for line in file.readlines() if line.strip()]  # load all file lines

            if len(lines) < 3:
                raise ValueError("File must have at least 4 lines")
            if lines[2] == '0':
                raise ValueError("File hase no samples for the signal")

            # Parse header
            signal_type = int(lines[0])
            is_periodic = int(lines[1])
            num_samples_header = int(lines[2])

            indices = []
            samples = []

            # Parse data lines
            for i in range(3, min(3 + num_samples_header, len(lines))):
                parts = lines[i].split()

                if len(parts) >= 2:  # check it hase split correct

                    index_str = parts[0].rstrip('f')
                    sample_str = parts[1].rstrip('f')

                    indices.append(float(index_str))
                    samples.append(float(sample_str))

            # return Dictionary after filtering the data
            return {
                "indices": np.array(indices),
                "samples": np.array(samples),
                "signal_type": signal_type,
                "is_periodic": is_periodic,
                "num_samples": len(samples)
            }
        # for errors only 😑
        except Exception as e:
            raise ValueError(f"Error parsing file: {str(e)}")

    @staticmethod
    def add_signals(signals):
        """Add multiple signals (handles unequal lengths by zero-padding)"""
        if len(signals) < 2:
            raise ValueError("Need at least 2 signals to perform addition")

        # If signals have different indices, will take the max indices
        longest_indices = max(signals, key=lambda s: len(s["indices"]))["indices"]

        # Find the maximum signal sample length
        max_len = max(len(sig["samples"]) for sig in signals)

        result_samples = [0.0] * max_len
        # Perform addition with zero-padding
        for sig in signals:
            for i in range(max_len):
                if i < len(sig["samples"]):
                    result_samples[i] += sig["samples"][i]
                else:
                    result_samples[i] += 0  # Explicitly showing zero-padding

        # return the resulting signal dictionary
        return {
            "indices": np.array(longest_indices[:max_len]),
            "samples": np.array(result_samples),
            "signal_type": 0,
            "is_periodic": 0,
            "num_samples": max_len
        }

    @staticmethod
    def subtract_signals(signal1, signal2):
        """Subtract signal2 from signal1 with proper index alignment"""

        # Get all unique indices from both signals
        all_indices = sorted(set(signal1["indices"]) | set(signal2["indices"]))

        # Create dictionaries for quick lookup
        samples1_dict = dict(zip(signal1["indices"], signal1["samples"]))
        samples2_dict = dict(zip(signal2["indices"], signal2["samples"]))

        result_samples = []
        result_indices = []

        # Perform subtraction with proper alignment
        for index in all_indices:
            val1 = samples1_dict.get(index, 0.0)  # Default to 0 if index not found
            val2 = samples2_dict.get(index, 0.0)  # Default to 0 if index not found
            result_samples.append(val1 - val2)
            result_indices.append(index)

        return {
            "indices": np.array(result_indices),
            "samples": np.array(result_samples),
            "signal_type": 0,
            "is_periodic": 0,
            "num_samples": len(result_samples)
        }

    @staticmethod
    def multiply_signal(signal, constant):
        """Multiply signal by constant"""
        return {
            "indices": signal["indices"],
            "samples": signal['samples'] * constant,  # Multiply each sample with number
            "signal_type": signal["signal_type"],
            "is_periodic": signal["is_periodic"],
            "num_samples": signal["num_samples"]
        }

    @staticmethod
    def square_signal(signal):
        """Square signal samples"""
        return {
            "indices": signal["indices"],
            "samples": signal["samples"] ** 2,  # square the sampels array
            "signal_type": signal["signal_type"],
            "is_periodic": 0,
            "num_samples": signal["num_samples"]
        }

    @staticmethod
    def normalize_signal(signal, target_range):
        """Normalize signal to range"""
        samples = signal["samples"]
        min_val = np.min(samples)
        max_val = np.max(samples)

        if max_val == min_val:
            normalized = [0] * len(samples)
        else:
            normalized = [(s - min_val) / (max_val - min_val) for s in samples]

            if target_range == "-1_to_1":
                normalized = [2 * n - 1 for n in normalized]

        return {
            "indices": signal["indices"],
            "samples": np.array(normalized),
            "signal_type": signal["signal_type"],
            "is_periodic": signal["is_periodic"],
            "num_samples": signal["num_samples"]
        }

    @staticmethod
    def accumulate_signal(signal):
        """Accumulate signal (running sum)"""
        samples = signal["samples"]
        accumulated = []
        total = 0

        for s in samples:
            total += s
            accumulated.append(total)

        return {
            "indices": signal["indices"],
            "samples": accumulated,
            "signal_type": signal["signal_type"],
            "is_periodic": 0,
            "num_samples": signal["num_samples"]
        }

    # Signal Generation Methods
    @staticmethod
    def generate_sinusoidal(amplitude, analog_freq, phase_deg, sampling_freq, is_cosine=False):
        """Generate sine/cosine wave - phase input in DEGREES"""
        # Validate sampling theorem
        # if sampling_freq <= 2 * analog_freq:
        #     raise ValueError(f"Sampling frequency must be > {2 * analog_freq} Hz")

        # Convert phase from degrees to radians
        phase_rad = phase_deg

        # Calculate number of samples for one second (standard practice)
        num_samples = int(sampling_freq)  # 1 second worth of samples

        # Create sample indices (0, 1, 2, ..., num_samples-1)
        indices = np.arange(num_samples)

        # Calculate time values for each sample
        t = indices / sampling_freq

        # Generate wave using numpy functions
        if is_cosine:
            samples = amplitude * np.cos(2 * np.pi * analog_freq * t + phase_rad)
        else:
            samples = amplitude * np.sin(2 * np.pi * analog_freq * t + phase_rad)

        return {
            "indices": indices,  # FIX: Use integer indices (0, 1, 2, ...) not time values
            "samples": samples,
            "signal_type": 0,
            "is_periodic": 1,
            "num_samples": num_samples,
            "x_label": "Sample Index (n)"
        }

    @staticmethod
    def quantize_signal(signal, levels=None, bits=None):

        samples = signal["samples"]
        indices = signal["indices"]  # for the display method only

        # Calculate levels from bits if bits provided
        if bits is not None:
            levels = 2 ** bits
        elif levels is not None:
            bits = math.ceil(math.log2(levels))
            # ceiling the fractions
        else:
            raise ValueError("Either levels or bits must be provided")

        # Find min and  max of signal
        min_val = np.min(samples)
        max_val = np.max(samples)

        # Calculate step size "the size of every level "
        step = (max_val - min_val) / levels

        quantized_samples = []
        binary_codes = []
        interval_indices = [] # display only
        error_samples = []

        for sample in samples:
            # find which interval the sample is in
            level = int((sample - min_val) / step)

            # Handle edge case where sample equals max_val
            if level >= levels:
                level = levels - 1

            # Ensure level is within valid range
            level = max(0, min(levels - 1, level))

            # Quantized value at the midpoint of the interval
            quantized_value = min_val + (level + 0.5) * step

            # Generate binary code with exactly 'bits' bits
            binary_code = format(level, f'0{bits}b')

            quantized_samples.append(quantized_value)
            binary_codes.append(binary_code)
            interval_indices.append(level + 1)  # 1-based for display
            error_samples.append(quantized_value - sample)  # Calculate error

        return {
            "original_signal": signal,
            "quantized_signal": {
                "indices": indices,
                "samples": np.array(quantized_samples),
                "signal_type": signal["signal_type"],
                "is_periodic": signal["is_periodic"],
                "num_samples": signal["num_samples"]
            },
            "error_signal": {
                "indices": indices,
                "samples": np.array(error_samples),  # Use calculated error samples
                "signal_type": signal["signal_type"],
                "is_periodic": signal["is_periodic"],
                "num_samples": signal["num_samples"]
            },
            "binary_codes": binary_codes,
            "interval_indices": np.array(interval_indices),
            "quantization_info": {
                "levels": levels,
                "bits": bits,
                "min_val": min_val,
                "max_val": max_val,
                "step_size": step
            }
        }

    @staticmethod
    def calculate_quantization_metrics(original_signal, quantized_signal):
        """Calculate quantization performance metrics"""
        original = original_signal["samples"]
        quantized = quantized_signal["samples"]
        error = original - quantized

        mse = np.mean(error ** 2)
        rmse = np.sqrt(mse)
        signal_power = np.mean(original ** 2)
        snr = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')

        return {
            "mse": mse,
            "rmse": rmse,
            "snr_db": snr,
            "max_error": np.max(np.abs(error)),
            "mean_error": np.mean(error)
        }

    #---------------------------
    #Task 3
    #----------------------------

class SignalDataManager:
    """Manages signal data storage"""

    def __init__(self):
        self.signals = []

    def add_signal(self, signal_data, name=""):
        """Add signal to manager"""
        signal_data["name"] = name
        self.signals.append(signal_data)
        return len(self.signals) - 1

    def get_signal(self, index):
        """Get signal by index"""
        return self.signals[index] if 0 <= index < len(self.signals) else None

    def get_all_signals(self):
        """Get all signals"""
        return self.signals

    def clear_signals(self):
        """Clear all signals"""
        self.signals = []

    def remove_signal(self, index):
        """Remove specific signal by index"""
        if 0 <= index < len(self.signals):
            removed_signal = self.signals.pop(index)
            return removed_signal
        return None

    def remove_signals(self, indices):
        """Remove multiple signals by indices (in descending order to avoid index shifting)"""
        # Sort indices in descending order to avoid index shifting during removal
        indices_sorted = sorted(indices, reverse=True)
        removed_signals = []

        for index in indices_sorted:
            if 0 <= index < len(self.signals):
                removed_signals.append(self.signals.pop(index))

        return removed_signals

    def get_signal_count(self):
        """Get signal count"""
        return len(self.signals)
