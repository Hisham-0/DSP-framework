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
        try:
            with open(file_path, 'r') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]

            if len(lines) < 3:
                raise ValueError("File must have at least 4 lines")

            # Parse header
            signal_type = int(lines[0])
            is_periodic = int(lines[1])
            num_samples_header = int(lines[2])

            indices = []
            samples = []

            for i in range(3, min(3 + num_samples_header, len(lines))):
                parts = lines[i].split()
                if len(parts) >= 2:
                    index_str = parts[0].rstrip('f')
                    sample_str = parts[1].rstrip('f')
                    indices.append(float(index_str))
                    samples.append(float(sample_str))

            # ENSURE NUMPY ARRAYS - this is key
            return {
                "indices": np.array(indices, dtype=float),
                "samples": np.array(samples, dtype=float),
                "signal_type": signal_type,
                "is_periodic": is_periodic,
                "num_samples": len(samples)
            }
        except Exception as e:
            raise ValueError(f"Error parsing file: {str(e)}")

    @staticmethod
    def add_signals(signals):
        if len(signals) < 2:
            raise ValueError("Need at least 2 signals to perform addition")

        # Use NumPy for all operations
        max_len = max(len(sig["samples"]) for sig in signals)

        # Initialize result with zeros using NumPy
        result_samples = np.zeros(max_len)

        for sig in signals:
            # Pad signal to max length using NumPy
            padded_samples = np.pad(sig["samples"], (0, max_len - len(sig["samples"])), 'constant')
            result_samples += padded_samples  # Vectorized addition

        return {
            "indices": np.arange(max_len),  # Proper NumPy indices
            "samples": result_samples,
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
        samples = signal["samples"]
        min_val = np.min(samples)
        max_val = np.max(samples)

        if max_val == min_val:
            normalized = np.zeros_like(samples)  # NumPy zeros
        else:
            normalized = (samples - min_val) / (max_val - min_val)  # Vectorized

            if target_range == "-1_to_1":
                normalized = 2 * normalized - 1  # Vectorized

        return {
            "indices": signal["indices"],
            "samples": normalized,
            "signal_type": signal["signal_type"],
            "is_periodic": signal["is_periodic"],
            "num_samples": signal["num_samples"]
        }

    @staticmethod
    def accumulate_signal(signal):
        """Use NumPy cumsum for accumulation"""
        return {
            "indices": signal["indices"],
            "samples": np.cumsum(signal["samples"]),  # Vectorized accumulation
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
        interval_indices = []  # display only
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

    # ---------------------------
    # Task 3
    # ----------------------------

    @staticmethod
    def compute_dft(signal, sampling_freq):
        """
        Compute Discrete Fourier Transform manually without using np.fft

        DFT Formula: X(k) = Σ(n=0 to N-1) x(n) * e^(-j*2π*k*n/N)
        where:
        - X(k) is the k-th frequency component
        - x(n) is the n-th time domain sample
        - N is the total number of samples
        - j is the imaginary unit (sqrt(-1))
        - k is the frequency bin index (0 to N-1)
    """
        samples = signal["samples"]
        N = len(samples)

        # Initialize array to store complex DFT components
        dft_components = np.zeros(N, dtype=complex)

        # For each frequency bin k (0 to N-1)
        for k in range(N):
            # Initialize the k-th frequency component as complex number
            real_part = 0.0
            imag_part = 0.0

            # Sum over all time domain samples n (0 to N-1)
            for n in range(N):
                # Calculate the angle: -2π*k*n/N
                angle = -2.0 * np.pi * k * n / N

                # Using Euler's formula: e^(jθ) = cos(θ) + j*sin(θ)
                # For e^(-jθ) = cos(θ) - j*sin(θ)
                cos_component = np.cos(angle)
                sin_component = np.sin(angle)

                real_part += samples[n] * cos_component
                imag_part += samples[n] * (-sin_component)

            # Store the complex result for this frequency bin
            dft_components[k] = complex(real_part, imag_part)

        amplitudes = np.abs(dft_components) / N * 2
        amplitudes[0] /= 2  # DC component (k=0) is not doubled

        # Calculate phases from complex components
        # Phase = angle(X(k)) = arctan(imag/real)
        phases = np.angle(dft_components)

        frequencies = np.arange(N) * sampling_freq / N

        return {
            'components': dft_components,
            'amplitudes': amplitudes,
            'phases': phases,
            'frequencies': frequencies
        }

    # @staticmethod
    # def compute_idft(dft_components, num_samples, original_indices):
    #     """
    #     Compute Inverse Discrete Fourier Transform manually without using np.fft
    #         """
    #     N = num_samples
    #
    #     # Initialize array to store reconstructed time domain samples
    #     reconstructed_samples = np.zeros(N, dtype=complex)
    #
    #     # For each time sample n (0 to N-1)
    #     for n in range(N):
    #         # Initialize the n-th time sample as complex number
    #         real_part = 0.0
    #         imag_part = 0.0
    #
    #         # Sum over all frequency components k (0 to N-1)
    #         for k in range(N):
    #             # Calculate the angle: +2π*k*n/N (note: positive for IDFT)
    #             angle = 2.0 * np.pi * k * n / N
    #
    #             # Using Euler's formula: e^(jθ) = cos(θ) + j*sin(θ)
    #             cos_component = np.cos(angle)
    #             sin_component = np.sin(angle)
    #
    #             # Get the k-th frequency component (complex number)
    #             X_k_real = dft_components[k].real
    #             X_k_imag = dft_components[k].imag
    #
    #             real_part += X_k_real * cos_component - X_k_imag * sin_component
    #             imag_part += X_k_real * sin_component + X_k_imag * cos_component
    #
    #         # Apply normalization factor (1/N) and store result
    #         reconstructed_samples[n] = complex(real_part / N, imag_part / N)
    #
    #     # Return only the real part since original signal was real
    #     # (imaginary part should be near zero due to numerical precision)
    #     return original_indices, np.real(reconstructed_samples)

    @staticmethod
    def compute_idft(dft_components, num_samples=None, original_indices=None):
        """
        Compute Inverse Discrete Fourier Transform manually without using np.fft

        Args:
            dft_components: Complex DFT components array
            num_samples: Number of samples in the reconstructed signal
            original_indices: Original time indices for the reconstructed signal

        Returns:
            tuple: (time_indices, time_domain_samples) - The reconstructed signal in time domain
        """
        N = len(dft_components) if num_samples is None else num_samples

        # Initialize array to store reconstructed time domain samples
        reconstructed_samples = np.zeros(N, dtype=complex)

        # For each time sample n (0 to N-1)
        for n in range(N):
            # Initialize the n-th time sample as complex number
            real_part = 0.0
            imag_part = 0.0

            # Sum over all frequency components k (0 to N-1)
            for k in range(len(dft_components)):
                # Calculate the angle: +2π*k*n/N (note: positive for IDFT)
                angle = 2.0 * np.pi * k * n / N

                # Using Euler's formula: e^(jθ) = cos(θ) + j*sin(θ)
                cos_component = np.cos(angle)
                sin_component = np.sin(angle)

                # Get the k-th frequency component (complex number)
                X_k_real = dft_components[k].real
                X_k_imag = dft_components[k].imag

                real_part += X_k_real * cos_component - X_k_imag * sin_component
                imag_part += X_k_real * sin_component + X_k_imag * cos_component

            # Apply normalization factor (1/N) and store result
            reconstructed_samples[n] = complex(real_part / N, imag_part / N)

        # Generate time indices if not provided
        if original_indices is None:
            time_indices = np.arange(N)
        else:
            time_indices = original_indices

        # Return only the real part since original signal was real
        # (imaginary part should be near zero due to numerical precision)
        print(f"{time_indices } {np.real(reconstructed_samples)} \n")
        return time_indices, np.real(reconstructed_samples)

    @staticmethod
    def remove_dc_component(signal):
        """Remove DC component from signal"""
        samples = signal["samples"]
        dc_removed_samples = samples - np.mean(samples)

        return {
            "indices": signal["indices"],
            "samples": dc_removed_samples,
            "signal_type": signal["signal_type"],
            "is_periodic": signal["is_periodic"],
            "num_samples": signal["num_samples"]
        }

    @staticmethod
    def get_dft_plot_data(dft_components, sampling_freq):
        """Calculate frequencies, normalized amplitudes, and phases for plotting"""
        N = len(dft_components)
        frequencies = np.arange(N) * sampling_freq / N

        # Get amplitudes and normalize
        amplitudes = np.abs(dft_components)
        normalized_amplitudes = amplitudes / N * 2
        normalized_amplitudes[0] /= 2  # DC component is not doubled

        # Clamp near-zero values
        normalized_amplitudes[normalized_amplitudes < 1e-10] = 0

        phases = np.angle(dft_components)
        # Set phase to 0 where amplitude is 0
        phases[normalized_amplitudes == 0] = 0

        return frequencies, normalized_amplitudes, phases


class SignalDataManager:
    """Manages signal data storage"""

    def __init__(self):
        self.signals = []

    def add_signal(self, signal_data, name=""):
        """Add signal to manager - ensure samples and indices are NumPy arrays"""
        signal_data["name"] = name

        # Convert to NumPy arrays if they aren't already
        if not isinstance(signal_data["indices"], np.ndarray):
            signal_data["indices"] = np.array(signal_data["indices"])
        if not isinstance(signal_data["samples"], np.ndarray):
            signal_data["samples"] = np.array(signal_data["samples"])

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
