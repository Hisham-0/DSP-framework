"""
Signal Processing Framework - Main Entry Point
==================================================
This is the main application file that initializes and runs the signal processing GUI.
"""

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import os
import sys
from datetime import datetime
import numpy as np
from GUI.dft_window import DFTWindow
from GUI.left_panel import LeftPanel
from GUI.plot_area import PlotArea

# Import business logic from Data folder
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from Data.signal_processor import SignalProcessor, SignalDataManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the Data.signal_processor module is available")
    sys.exit(1)


class MenuBar:
    """Application Menu Bar"""

    def __init__(self, root, gui):
        self.root = root
        self.gui = gui
        self._create_menu()

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File Menu
        self._create_file_menu(menubar)

        # Signal Generation Menu
        self._create_generation_menu(menubar)

        # Operations Menu
        self._create_operations_menu(menubar)

        # frequency menu
        self._create_frequency_domain_menu(menubar)

        # Display Menu
        self._create_display_menu(menubar)

        # Test menu
        self._create_test_menu(menubar)

    def _create_file_menu(self, menubar):
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="Load Signal", command=self.gui.load_signal)
        file_menu.add_separator()
        file_menu.add_command(
            label="Delete Selected Signal",
            command=self.gui.delete_signal
        )
        file_menu.add_command(
            label="Delete All Signals",
            command=self.gui.delete_all_signals
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

    def _create_generation_menu(self, menubar):
        gen_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Generation", menu=gen_menu)

        gen_menu.add_command(
            label="Sine Wave",
            command=lambda: SinusoidalDialog(self.root, "Sine", self.gui).show()
        )
        gen_menu.add_command(
            label="Cosine Wave",
            command=lambda: SinusoidalDialog(self.root, "Cosine", self.gui).show()
        )

    def _create_operations_menu(self, menubar):
        ops_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Operations", menu=ops_menu)

        ops_menu.add_command(
            label="Addition",
            command=lambda: self._show_operation_dialog("Add Signals", "addition")
        )
        ops_menu.add_command(
            label="Subtraction",
            command=lambda: self._show_operation_dialog("Subtract Signals", "subtraction")
        )
        ops_menu.add_command(
            label="Multiplication",
            command=lambda: self._show_operation_dialog("Multiply Signal", "multiplication")
        )
        ops_menu.add_command(
            label="Squaring",
            command=lambda: self._show_operation_dialog("Square Signal", "squaring")
        )
        ops_menu.add_separator()
        ops_menu.add_command(
            label="Normalization",
            command=lambda: self._show_operation_dialog("Normalize Signal", "normalization")
        )
        ops_menu.add_command(
            label="Accumulation",
            command=lambda: self._show_operation_dialog("Accumulate Signal", "accumulation")
        )

    def _create_display_menu(self, menubar):
        display_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Display", menu=display_menu)

        display_menu.add_command(
            label="Reset View",
            command=self.gui.reset_plot_view
        )

    def _create_test_menu(self, menubar):
        """Create the Test menu"""
        test_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Test", menu=test_menu)

        test_menu.add_command(
            label="Run Tests",
            command=lambda: TestDialog(self.root, self.gui).show()
        )

    def _create_frequency_domain_menu(self, menubar):
        """Create Frequency Domain menu"""
        freq_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Frequency Domain", menu=freq_menu)

        freq_menu.add_command(
            label="Fourier Transform",
            command=self.gui.show_fourier_transform_dialog
        )
        freq_menu.add_separator()
        freq_menu.add_command(
            label="Run Fourier Tests",
            command=lambda: FourierTestDialog(self.root, self.gui).show()
        )

    def _show_operation_dialog(self, title, operation_type):
        count = self.gui.data_manager.get_signal_count()

        if operation_type in ["addition", "subtraction"] and count < 2:
            messagebox.showwarning(
                "Warning",
                f"Need at least 2 signals for {operation_type}"
            )
            return
        elif count < 1:
            messagebox.showwarning("Warning", "No signals loaded")
            return

        OperationDialog(self.root, title, self.gui, operation_type).show()


class SinusoidalDialog:
    """Sinusoidal Wave Generation Dialog"""

    def __init__(self, parent, wave_type, gui):
        self.parent = parent
        self.wave_type = wave_type
        self.gui = gui
        self.dialog = None

    def show(self):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(f"Generate {self.wave_type} Wave")
        self.dialog.geometry("300x200")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        tk.Label(
            self.dialog,
            text=f"{self.wave_type} Wave Parameters",
            font=("Arial", 9, "bold")
        ).pack(pady=10)

        self.amplitude_var = self._create_parameter_field("Amplitude:", 1.0)
        self.freq_var = self._create_parameter_field("Frequency (Hz):", 1.0)
        self.phase_var = self._create_parameter_field("Phase (degrees):", 0.0)
        self.sampling_freq_var = self._create_parameter_field("Sampling Freq (Hz):", 100.0)

        self._create_buttons()

    def _create_parameter_field(self, label, default):
        frame = tk.Frame(self.dialog)
        frame.pack(fill=tk.X, padx=10, pady=2)

        tk.Label(
            frame,
            text=label,
            width=15,
            anchor='w'
        ).pack(side=tk.LEFT)

        var = tk.StringVar(value=str(default))
        tk.Entry(
            frame,
            textvariable=var,
            width=10
        ).pack(side=tk.RIGHT)

        return var

    def _create_buttons(self):
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="Generate",
            command=self._generate,
            width=10
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="Cancel",
            command=self.dialog.destroy,
            width=10
        ).pack(side=tk.LEFT, padx=5)

    def _generate(self):
        try:
            amplitude = float(self.amplitude_var.get())
            analog_freq = float(self.freq_var.get())
            phase_deg = float(self.phase_var.get())
            sampling_freq = float(self.sampling_freq_var.get())
            is_cosine = (self.wave_type == "Cosine")

            if any(v <= 0 for v in [sampling_freq, analog_freq]):
                raise ValueError("Frequency and sampling rate must be positive.")

            params = {
                'amplitude': amplitude,
                'analog_freq': analog_freq,
                'phase_deg': phase_deg,
                'sampling_freq': sampling_freq,
                'is_cosine': is_cosine
            }

            success = self.gui.execute_operation("sinusoidal", **params)

            if success:
                self.dialog.destroy()

        except ValueError as e:
            messagebox.showerror("Error", f"Input Error:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Generation failed:\n{str(e)}")


class OperationDialog:
    """Generic Signal Operation Dialog"""

    def __init__(self, parent, title, gui, operation_type):
        self.parent = parent
        self.title = title
        self.gui = gui
        self.operation_type = operation_type
        self.dialog = None

    def show(self):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(self.title)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        if self.operation_type in ["addition", "subtraction"]:
            self.dialog.geometry("300x400")
        else:
            self.dialog.geometry("250x300")

        self._create_content()
        self._create_buttons()

    def _create_content(self):
        tk.Label(
            self.dialog,
            text=self.title,
            font=("Arial", 9, "bold")
        ).pack(pady=10)

        if self.operation_type == "addition":
            self._create_addition_content()
        elif self.operation_type == "subtraction":
            self._create_subtraction_content()
        elif self.operation_type == "multiplication":
            self._create_multiplication_content()
        elif self.operation_type in ["squaring", "normalization", "accumulation"]:
            self._create_single_signal_content()

    def _create_quantization_content(self):
        tk.Label(self.dialog, text="Select signal:").pack(pady=5)
        self.signal_list = tk.Listbox(self.dialog, height=4, exportselection=False)
        self.signal_list.pack(padx=10, pady=2, fill=tk.X)
        self._populate_signal_list([self.signal_list])

        # Quantization parameters
        param_frame = tk.Frame(self.dialog)
        param_frame.pack(fill=tk.X, padx=10, pady=10)

        self.quant_method = tk.StringVar(value="levels")

        tk.Radiobutton(
            param_frame,
            text="Levels:",
            variable=self.quant_method,
            value="levels"
        ).grid(row=0, column=0, sticky='w')

        self.levels_var = tk.StringVar(value="8")
        tk.Entry(param_frame, textvariable=self.levels_var, width=8).grid(row=0, column=1, padx=5)

        tk.Radiobutton(
            param_frame,
            text="Bits:",
            variable=self.quant_method,
            value="bits"
        ).grid(row=1, column=0, sticky='w', pady=5)

        self.bits_var = tk.StringVar(value="3")
        tk.Entry(param_frame, textvariable=self.bits_var, width=8).grid(row=1, column=1, padx=5)

    def _create_addition_content(self):
        tk.Label(self.dialog, text="Select signals to add:").pack(pady=5)

        frame = tk.Frame(self.dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            frame,
            selectmode=tk.MULTIPLE,
            yscrollcommand=scrollbar.set,
            height=8
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        self._populate_signal_list()

    def _create_subtraction_content(self):
        tk.Label(self.dialog, text="Signal 1 (Subtrahend):").pack(pady=5)
        self.signal1_list = tk.Listbox(self.dialog, height=4, exportselection=False)
        self.signal1_list.pack(padx=10, pady=2, fill=tk.X)

        tk.Label(self.dialog, text="Signal 2 (Minuend):").pack(pady=5)
        self.signal2_list = tk.Listbox(self.dialog, height=4, exportselection=False)
        self.signal2_list.pack(padx=10, pady=2, fill=tk.X)

        self._populate_signal_list([self.signal1_list, self.signal2_list])

    def _create_multiplication_content(self):
        tk.Label(self.dialog, text="Select signal:").pack(pady=5)
        self.signal_list = tk.Listbox(self.dialog, height=4, exportselection=False)
        self.signal_list.pack(padx=10, pady=2, fill=tk.X)
        self._populate_signal_list([self.signal_list])

        tk.Label(self.dialog, text="Enter constant:").pack(pady=5)
        self.constant_var = tk.StringVar(value="1.0")
        tk.Entry(
            self.dialog,
            textvariable=self.constant_var,
            width=15
        ).pack(pady=2)

    def _create_single_signal_content(self):
        tk.Label(self.dialog, text="Select signal:").pack(pady=5)
        self.signal_list = tk.Listbox(self.dialog, height=4, exportselection=False)
        self.signal_list.pack(padx=10, pady=2, fill=tk.X)
        self._populate_signal_list([self.signal_list])

        if self.operation_type == "normalization":
            tk.Label(
                self.dialog,
                text="Target Range:",
                font=("Arial", 8, "bold")
            ).pack(pady=5)

            self.range_var = tk.StringVar(value="-1_to_1")

            tk.Radiobutton(
                self.dialog,
                text="-1 to 1",
                variable=self.range_var,
                value="-1_to_1"
            ).pack()

            tk.Radiobutton(
                self.dialog,
                text="0 to 1",
                variable=self.range_var,
                value="0_to_1"
            ).pack()

    def _populate_signal_list(self, listboxes=None):
        if listboxes is None:
            listboxes = [self.listbox] if hasattr(self, 'listbox') else []

        for listbox in listboxes:
            listbox.delete(0, tk.END)
            for i, signal in enumerate(self.gui.data_manager.get_all_signals()):
                listbox.insert(tk.END, f"{i + 1}. {signal['name']}")

    def _create_buttons(self):
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="Process",
            command=self._process,
            width=10
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="Cancel",
            command=self.dialog.destroy,
            width=10
        ).pack(side=tk.LEFT, padx=5)

    def _process(self):
        try:
            if self.operation_type == "addition":
                selected = self.listbox.curselection()
                if len(selected) < 2:
                    messagebox.showwarning("Warning", "Select at least 2 signals")
                    return
                self.gui.execute_operation("addition", signal_indices=selected)

            elif self.operation_type == "subtraction":
                sel1 = self.signal1_list.curselection()
                sel2 = self.signal2_list.curselection()
                if not sel1 or not sel2:
                    messagebox.showwarning("Warning", "Select both signals")
                    return
                if sel1[0] == sel2[0]:
                    messagebox.showwarning("Warning", "Select different signals")
                    return
                self.gui.execute_operation(
                    "subtraction",
                    signal1_index=sel2[0],
                    signal2_index=sel1[0]
                )

            elif self.operation_type == "multiplication":
                selected = self.signal_list.curselection()
                if not selected:
                    messagebox.showwarning("Warning", "Select a signal")
                    return
                constant = float(self.constant_var.get())
                self.gui.execute_operation(
                    "multiplication",
                    signal_index=selected[0],
                    constant=constant
                )

            elif self.operation_type == "squaring":
                selected = self.signal_list.curselection()
                if not selected:
                    messagebox.showwarning("Warning", "Select a signal")
                    return
                self.gui.execute_operation("squaring", signal_index=selected[0])

            elif self.operation_type == "normalization":
                selected = self.signal_list.curselection()
                if not selected:
                    messagebox.showwarning("Warning", "Select a signal")
                    return
                self.gui.execute_operation(
                    "normalization",
                    signal_index=selected[0],
                    target_range=self.range_var.get()
                )

            elif self.operation_type == "accumulation":
                selected = self.signal_list.curselection()
                if not selected:
                    messagebox.showwarning("Warning", "Select a signal")
                    return
                self.gui.execute_operation("accumulation", signal_index=selected[0])

            self.dialog.destroy()

        except ValueError as e:
            messagebox.showerror("Error", f"Input error:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Operation failed:\n{str(e)}")


class QuantizationDialog:
    """Quantization Operation Dialog"""

    def __init__(self, parent, gui):
        self.parent = parent
        self.gui = gui
        self.dialog = None

    def show(self):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Quantize Signal")
        self.dialog.geometry("300x300")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        tk.Label(
            self.dialog,
            text="Signal Quantization",
            font=("Arial", 9, "bold")
        ).pack(pady=10)

        # Signal selection
        tk.Label(self.dialog, text="Select signal:").pack(pady=5)
        self.signal_list = tk.Listbox(self.dialog, height=4, exportselection=False)
        self.signal_list.pack(padx=10, pady=2, fill=tk.X)
        self._populate_signal_list()

        # Quantization method selection
        method_frame = tk.Frame(self.dialog)
        method_frame.pack(fill=tk.X, padx=10, pady=10)

        self.method_var = tk.StringVar(value="levels")

        tk.Radiobutton(
            method_frame,
            text="Number of Levels:",
            variable=self.method_var,
            value="levels",
            command=self._toggle_method
        ).pack(anchor='w')

        self.levels_frame = tk.Frame(method_frame)
        self.levels_frame.pack(fill=tk.X, padx=20)
        self.levels_var = tk.StringVar(value="8")
        tk.Entry(self.levels_frame, textvariable=self.levels_var, width=10).pack(side=tk.LEFT)

        tk.Radiobutton(
            method_frame,
            text="Number of Bits:",
            variable=self.method_var,
            value="bits",
            command=self._toggle_method
        ).pack(anchor='w', pady=(5, 0))

        self.bits_frame = tk.Frame(method_frame)
        self.bits_frame.pack(fill=tk.X, padx=20)
        self.bits_var = tk.StringVar(value="3")
        tk.Entry(self.bits_frame, textvariable=self.bits_var, width=10).pack(side=tk.LEFT)

        # Buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="Quantize",
            command=self._quantize,
            width=10
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="Cancel",
            command=self.dialog.destroy,
            width=10
        ).pack(side=tk.LEFT, padx=5)

        self._toggle_method()

    def _populate_signal_list(self):
        self.signal_list.delete(0, tk.END)
        for i, signal in enumerate(self.gui.data_manager.get_all_signals()):
            self.signal_list.insert(tk.END, f"{i + 1}. {signal['name']}")

    def _toggle_method(self):
        if self.method_var.get() == "levels":
            self.levels_var.set("8")
        else:
            self.bits_var.set("3")

    def _quantize(self):
        try:
            selected = self.signal_list.curselection()
            if not selected:
                messagebox.showwarning("Warning", "Select a signal")
                return

            method = self.method_var.get()
            if method == "levels":
                levels = int(self.levels_var.get())
                bits = None
                if levels < 2:
                    raise ValueError("Number of levels must be at least 2")
            else:
                bits = int(self.bits_var.get())
                levels = None
                if bits < 1:
                    raise ValueError("Number of bits must be at least 1")

            self.gui.execute_operation(
                "quantization",
                signal_index=selected[0],
                levels=levels,
                bits=bits
            )
            self.dialog.destroy()

        except ValueError as e:
            messagebox.showerror("Error", f"Input error:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Quantization failed:\n{str(e)}")


# ---------------------------
# Task 3
# ----------------------------


# --------------------------------------------


class SignalProcessingGUI:
    """Main GUI Controller Class"""

    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processing Framework")
        self.root.state('zoomed')

        self.data_manager = SignalDataManager()
        self.signal_processor = SignalProcessor
        self._setup_output_folder()
        self._build_interface()

    def _setup_output_folder(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.output_folder = os.path.join(parent_dir, "generated_signals")
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output folder: {self.output_folder}")

    def _build_interface(self):
        self.menu_bar = MenuBar(self.root, self)
        self.left_panel = LeftPanel(self.root, self)
        self.plot_area = PlotArea(self.root, self)

    def load_signal(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        initial_dir = os.path.join(parent_dir, "signals")

        file_path = filedialog.askopenfilename(
            title="Select Signal File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=initial_dir if os.path.exists(initial_dir) else None
        )

        if file_path:
            try:
                signal_data = SignalProcessor.parse_signal_file(file_path)
                signal_name = os.path.basename(file_path)
                signal_data["generated"] = False
                signal_data["file_path"] = file_path
                self.data_manager.add_signal(signal_data, signal_name)
                self.left_panel.update_signal_list()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load signal:\n{str(e)}")

    def save_signal_to_file(self, signal_data, filename):
        try:
            file_path = os.path.join(self.output_folder, filename)
            with open(file_path, 'w') as f:
                f.write(f"{signal_data['signal_type']}\n")
                f.write(f"{signal_data['is_periodic']}\n")
                f.write(f"{signal_data['num_samples']}\n")
                for i in range(len(signal_data['indices'])):
                    index = int(signal_data['indices'][i])
                    value = signal_data['samples'][i]
                    f.write(f"{index} {value:.6f}\n")
            print(f"Signal saved to: {file_path}")
            return True
        except Exception as e:
            print(f"Error saving signal: {e}")
            messagebox.showerror("Save Error", f"Failed to save signal:\n{str(e)}")
            return False

    def save_quantization_test_file(self, quantization_result, filename):
        """Save quantization results in the format expected by test cases"""
        try:
            file_path = os.path.join(self.output_folder, filename)
            with open(file_path, 'w') as f:
                # Write header matching test case format
                f.write("0\n")  # signal_type
                f.write("0\n")  # is_periodic
                f.write(f"{quantization_result['quantized_signal']['num_samples']}\n")  # num_samples

                # Write data in format: [interval_index] [binary_code] [quantized_value] [error]
                indices = quantization_result['quantized_signal']['indices']
                samples = quantization_result['quantized_signal']['samples']
                binary_codes = quantization_result['binary_codes']
                error_samples = quantization_result['error_signal']['samples']
                interval_indices = quantization_result['interval_indices']

                for i in range(len(indices)):
                    f.write(
                        f"{binary_codes[i]} {samples[i]:.6f} {error_samples[i]:.6f}\n")  # {interval_indices[i]}  {error_samples[i]:.6f}

            print(f"Quantization test file saved to: {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving quantization test file: {e}")
            return False

    def delete_signal(self):
        selected_indices = self.left_panel.get_selected_signals()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a signal to delete")
            return
        if self.data_manager.get_signal_count() == 0:
            messagebox.showinfo("Info", "No signals to delete")
            return

        signal = self.data_manager.get_signal(selected_indices[0])
        response = messagebox.askyesno(
            "Delete Signal",
            f"Delete signal: {signal['name']}?"
        )

        if response:
            self.data_manager.remove_signals(selected_indices)
            self.left_panel.update_signal_list()
            self.update_plot_display()
            messagebox.showinfo("Success", "Signal deleted successfully")

    def delete_all_signals(self):
        if self.data_manager.get_signal_count() > 0:
            response = messagebox.askyesno(
                "Delete All Signals",
                "Are you sure you want to delete all signals?"
            )
            if response:
                self.data_manager.clear_signals()
                self.left_panel.update_signal_list()
                self.plot_area.clear_plot()
                messagebox.showinfo("Success", "All signals deleted")

    def update_plot_display(self):
        display_modes = self.left_panel.get_display_modes()
        self.plot_area.plot_signals(display_modes)

    def reset_plot_view(self):
        self.plot_area.reset_plot_view()

    # -----------------------------------------------------------
    # Operations call from the logic layer
    # -----------------------------------------------------------

    def execute_operation(self, operation_type, **kwargs):
        try:
            result = None
            name = None

            if operation_type == "sinusoidal":
                amplitude = kwargs.get('amplitude', 1.0)
                analog_freq = kwargs.get('analog_freq', 1.0)
                phase_deg = kwargs.get('phase_deg', 0.0)
                sampling_freq = kwargs.get('sampling_freq', 100.0)
                is_cosine = kwargs.get('is_cosine', False)

                result = SignalProcessor.generate_sinusoidal(
                    amplitude=amplitude,
                    analog_freq=analog_freq,
                    phase_deg=phase_deg,
                    sampling_freq=sampling_freq,
                    is_cosine=is_cosine
                )

                wave_type = "Cosine" if is_cosine else "Sine"
                name = f"{wave_type}_{amplitude}A_{analog_freq}Hz_{sampling_freq}Fs"

            elif operation_type == "addition":
                signal_indices = kwargs['signal_indices']
                signals = [
                    self.data_manager.get_signal(i)
                    for i in signal_indices
                ]
                result = SignalProcessor.add_signals(signals)
                name = "Addition_Result"

            elif operation_type == "subtraction":
                sig1 = self.data_manager.get_signal(kwargs['signal1_index'])
                sig2 = self.data_manager.get_signal(kwargs['signal2_index'])
                result = SignalProcessor.subtract_signals(sig1, sig2)
                name = "Subtraction_Result"

            elif operation_type == "multiplication":
                signal = self.data_manager.get_signal(kwargs['signal_index'])
                constant = kwargs['constant']
                result = SignalProcessor.multiply_signal(signal, constant)
                name = f"Multiplied_{constant}x"

            elif operation_type == "squaring":
                signal = self.data_manager.get_signal(kwargs['signal_index'])
                result = SignalProcessor.square_signal(signal)
                name = f"Squared_{signal['name']}"

            elif operation_type == "normalization":
                signal = self.data_manager.get_signal(kwargs['signal_index'])
                target_range = kwargs['target_range']
                result = SignalProcessor.normalize_signal(signal, target_range)
                name = f"Normalized_{signal['name']}"

            elif operation_type == "accumulation":
                signal = self.data_manager.get_signal(kwargs['signal_index'])
                result = SignalProcessor.accumulate_signal(signal)
                name = f"Accumulated_{signal['name']}"

            elif operation_type == "quantization":

                signal_index = kwargs['signal_index']
                signal = self.data_manager.get_signal(signal_index)

                # Ensure we're working with NumPy arrays
                if not isinstance(signal['samples'], np.ndarray):
                    signal['samples'] = np.array(signal['samples'])
                if not isinstance(signal['indices'], np.ndarray):
                    signal['indices'] = np.array(signal['indices'])

                levels = kwargs.get('levels')
                bits = kwargs.get('bits')

                quantization_result = SignalProcessor.quantize_signal(
                    signal, levels=levels, bits=bits
                )

                # The quantized signal should already have NumPy arrays
                quantized_signal = quantization_result["quantized_signal"]

                # Save and parse to ensure consistent format
                test_filename = f"quantization_test_output_{datetime.now().strftime('%H_%M_%S')}.txt"
                file_path = self.save_quantization_test_file(quantization_result, test_filename)

                # This will convert to NumPy arrays in parse_signal_file
                parsed_signal = SignalProcessor.parse_signal_file(file_path)
                self.data_manager.add_signal(parsed_signal, os.path.basename(file_path))

            if result and name:
                self.data_manager.add_signal(result, name)
                filename = f"{name}.txt"
                save_success = self.save_signal_to_file(result, filename)
                self.left_panel.update_signal_list()

                if save_success:
                    messagebox.showinfo(
                        "Success",
                        f"Operation completed successfully\n"
                        f"Result saved as: {filename}"
                    )
                else:
                    messagebox.showinfo(
                        "Success",
                        "Operation completed successfully\n"
                        "(Note: File save failed)"
                    )
                return True


            else:
                messagebox.showerror("Error", "Operation produced no result")
                return False

        except KeyError as e:
            messagebox.showerror("Error", f"Missing parameter: {str(e)}")
            return False
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid value: {str(e)}")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Operation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            return False

    # ----------------------------------------
    # ----------------------------------------
    # ----------------------------------------

    # Dialog methods for left panel
    def show_addition_dialog(self):
        OperationDialog(self.root, "Add Signals", self, "addition").show()

    def show_subtraction_dialog(self):
        OperationDialog(self.root, "Subtract Signals", self, "subtraction").show()

    def show_multiplication_dialog(self):
        OperationDialog(self.root, "Multiply Signal", self, "multiplication").show()

    def show_squaring_dialog(self):
        OperationDialog(self.root, "Square Signal", self, "squaring").show()

    def show_normalization_dialog(self):
        OperationDialog(self.root, "Normalize Signal", self, "normalization").show()

    def show_accumulation_dialog(self):
        OperationDialog(self.root, "Accumulate Signal", self, "accumulation").show()

    def show_quantization_dialog(self):
        """Show quantization dialog"""
        QuantizationDialog(self.root, self).show()

    # ---------------------------
    # task 3
    # --------------------------
    def show_fourier_transform_dialog(self):
        """Show Fourier Transform dialog"""
        selected_indices = self.left_panel.get_selected_signals()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a signal")
            return

        if len(selected_indices) > 1:
            messagebox.showwarning("Warning", "Please select only one signal")
            return

        # Ask for sampling frequency
        sampling_freq = simpledialog.askfloat(
            "Sampling Frequency",
            "Enter sampling frequency (Hz):",
            parent=self.root,
            minvalue=0.1,
            initialvalue=100.0
        )

        if sampling_freq is None:
            return

        signal_index = selected_indices[0]
        signal = self.data_manager.get_signal(signal_index)

        # Open DFT window
        DFTWindow(self.root, self, signal, sampling_freq)


class TestDialog:
    """Dialog for running signal processing tests"""

    def __init__(self, parent, gui):
        self.parent = parent
        self.gui = gui
        self.dialog = None

    def show(self):
        """Display the test dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Signal Processing Tests")
        self.dialog.geometry("500x700")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Main frame
        main_frame = tk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        tk.Label(
            main_frame,
            text="Signal Processing Test Suite",
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        # Test selection
        test_frame = tk.LabelFrame(main_frame, text="Select Test", font=("Arial", 9, "bold"))
        test_frame.pack(fill=tk.X, pady=5)

        self.test_var = tk.StringVar(value="squaring")

        # Define tests
        tests = [
            ("Squaring Test", "squaring"),
            ("Accumulation Test", "accumulation"),
            ("Normalization Test", "normalization"),
            ("Subtraction Test", "subtraction"),
            ("Sinusoidal Test", "sinusoidal"),
            ("Addition Test", "addition"),
            ("Quantization Test 1", "quantization1"),
            ("Quantization Test 2", "quantization2"),
        ]
        for text, value in tests:
            tk.Radiobutton(
                test_frame,
                text=text,
                variable=self.test_var,
                value=value,
                font=("Arial", 8)
            ).pack(anchor='w', padx=5, pady=2)

        # Signal selection
        signal_frame = tk.LabelFrame(main_frame, text="Select Signal", font=("Arial", 9, "bold"))
        signal_frame.pack(fill=tk.X, pady=5)

        self.signal_listbox = tk.Listbox(signal_frame, height=4, exportselection=False)
        self.signal_listbox.pack(fill=tk.X, padx=5, pady=5)
        self._populate_signal_list()

        # For multi-signal operations
        self.multi_signal_frame = tk.LabelFrame(main_frame, text="Select Additional Signals", font=("Arial", 9, "bold"))

        # For addition (multiple signals)
        self.addition_frame = tk.Frame(self.multi_signal_frame)
        self.addition_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(self.addition_frame, text="Select signals to add:").pack(anchor='w')
        self.addition_listbox = tk.Listbox(self.addition_frame, height=4, selectmode=tk.MULTIPLE, exportselection=False)
        self.addition_listbox.pack(fill=tk.X, pady=2)

        # For subtraction (single second signal)
        self.subtraction_frame = tk.Frame(self.multi_signal_frame)
        self.subtraction_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(self.subtraction_frame, text="Select second signal:").pack(anchor='w')
        self.subtraction_listbox = tk.Listbox(self.subtraction_frame, height=4, exportselection=False)
        self.subtraction_listbox.pack(fill=tk.X, pady=2)

        # For normalization range
        self.range_frame = tk.LabelFrame(main_frame, text="Normalization Range", font=("Arial", 9, "bold"))
        self.range_var = tk.StringVar(value="-1_to_1")
        tk.Radiobutton(
            self.range_frame,
            text="-1 to 1",
            variable=self.range_var,
            value="-1_to_1"
        ).pack(anchor='w', padx=5, pady=2)
        tk.Radiobutton(
            self.range_frame,
            text="0 to 1",
            variable=self.range_var,
            value="0_to_1"
        ).pack(anchor='w', padx=5, pady=2)

        # For sinusoidal wave type
        self.sinusoidal_frame = tk.LabelFrame(main_frame, text="Sinusoidal Wave Type", font=("Arial", 9, "bold"))
        self.wave_type_var = tk.StringVar(value="Sine")
        tk.Radiobutton(
            self.sinusoidal_frame,
            text="Sine Wave",
            variable=self.wave_type_var,
            value="Sine"
        ).pack(anchor='w', padx=5, pady=2)
        tk.Radiobutton(
            self.sinusoidal_frame,
            text="Cosine Wave",
            variable=self.wave_type_var,
            value="Cosine"
        ).pack(anchor='w', padx=5, pady=2)

        # Quantization parameters frame
        self.quant_frame = tk.LabelFrame(main_frame, text="Quantization Parameters", font=("Arial", 9, "bold"))

        # Method selection
        method_frame = tk.Frame(self.quant_frame)
        method_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        self.quant_method_var = tk.StringVar(value="bits")
        tk.Radiobutton(method_frame, text="Bits", variable=self.quant_method_var, value="bits").pack(side=tk.LEFT,
                                                                                                     padx=5)
        tk.Radiobutton(method_frame, text="Levels", variable=self.quant_method_var, value="levels").pack(side=tk.LEFT,
                                                                                                         padx=5)

        # Value input
        value_frame = tk.Frame(self.quant_frame)
        value_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(value_frame, text="Value:").pack(side=tk.LEFT)
        self.quant_value_var = tk.StringVar(value="3")
        tk.Entry(value_frame, textvariable=self.quant_value_var, width=10).pack(side=tk.LEFT, padx=5)

        # Expected output file selection
        self.file_frame = tk.LabelFrame(main_frame, text="Expected Output File", font=("Arial", 9, "bold"))
        file_select_frame = tk.Frame(self.file_frame)
        file_select_frame.pack(fill=tk.X, padx=5, pady=5)
        self.file_path_var = tk.StringVar()
        self.file_entry = tk.Entry(
            file_select_frame,
            textvariable=self.file_path_var,
            state='readonly',
            width=40
        )
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.browse_button = tk.Button(
            file_select_frame,
            text="Browse",
            command=self._browse_file,
            width=8
        )
        self.browse_button.pack(side=tk.RIGHT, padx=5)

        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="Run Test",
            command=self._run_test,
            width=12
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="Cancel",
            command=self.dialog.destroy,
            width=12
        ).pack(side=tk.LEFT, padx=5)

        # Update UI based on test type
        self.test_var.trace('w', self._on_test_change)
        self._populate_all_signal_lists()
        self._on_test_change()

    def _populate_all_signal_lists(self):
        """Populate all signal listboxes"""
        self._populate_signal_list([self.signal_listbox])
        self._populate_signal_list([self.subtraction_listbox])
        self._populate_signal_list([self.addition_listbox])

    def _populate_signal_list(self, listboxes=None):
        """Populate signal listboxes"""
        if listboxes is None:
            listboxes = [self.signal_listbox]

        for listbox in listboxes:
            listbox.delete(0, tk.END)
            signals = self.gui.data_manager.get_all_signals()
            for i, signal in enumerate(signals):
                listbox.insert(tk.END, f"{i}: {signal['name']}")

    def _browse_file(self):
        """Browse for expected output file"""
        if self.browse_button['state'] == 'normal':  # Only browse if button is enabled
            file_path = filedialog.askopenfilename(
                title="Select Expected Output File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                self.file_path_var.set(file_path)

    def _on_test_change(self, *args):
        """Update UI based on selected test type"""
        test_type = self.test_var.get()

        # Hide all optional frames
        self.multi_signal_frame.pack_forget()
        self.range_frame.pack_forget()
        self.sinusoidal_frame.pack_forget()
        self.quant_frame.pack_forget()
        self.file_frame.pack_forget()

        # Show relevant frames
        if test_type == "subtraction":
            self.multi_signal_frame.pack(fill=tk.X, pady=5)
            self.subtraction_frame.pack(fill=tk.X, padx=5, pady=5)
            self.addition_frame.pack_forget()
            self.browse_button.config(state='disabled')
            self.file_entry.config(state='disabled')
        elif test_type == "addition":
            self.multi_signal_frame.pack(fill=tk.X, pady=5)
            self.addition_frame.pack(fill=tk.X, padx=5, pady=5)
            self.subtraction_frame.pack_forget()
            self.browse_button.config(state='normal')
            self.file_entry.config(state='readonly')
            self.file_frame.pack(fill=tk.X, pady=5)
        elif test_type == "normalization":
            self.range_frame.pack(fill=tk.X, pady=5)
            self.browse_button.config(state='disabled')
            self.file_entry.config(state='disabled')
        elif test_type == "sinusoidal":
            self.sinusoidal_frame.pack(fill=tk.X, pady=5)
            self.browse_button.config(state='normal')
            self.file_entry.config(state='readonly')
            self.file_frame.pack(fill=tk.X, pady=5)
        elif test_type in ["quantization1", "quantization2"]:
            self.quant_frame.pack(fill=tk.X, pady=5)
            self.browse_button.config(state='normal')
            self.file_entry.config(state='readonly')
            self.file_frame.pack(fill=tk.X, pady=5)
        else:
            # For other tests (squaring, accumulation, etc.)
            self.browse_button.config(state='normal')
            self.file_entry.config(state='readonly')
            self.file_frame.pack(fill=tk.X, pady=5)

    def _get_signal_file_name(self, signal_idx):
        """Get the original file name of a signal or create a proper mapping"""
        signal = self.gui.data_manager.get_signal(signal_idx)

        # If signal was loaded from a file, it might have the original path
        if 'file_path' in signal:
            return os.path.basename(signal['file_path'])

        # Check if we can extract from the name
        name = signal['name']

        # Map common signal names to what the test expects
        name_lower = name.lower()

        # Handle generated signals by mapping to expected test names
        if 'signal1' in name_lower or signal_idx == 0:
            return 'Signal1.txt'
        elif 'signal2' in name_lower or signal_idx == 1:
            return 'Signal2.txt'
        elif 'signal3' in name_lower or signal_idx == 2:
            return 'Signal3.txt'
        else:
            # For other signals, try to extract or use index-based naming
            if '.txt' in name:
                return name
            else:
                # Use the actual file names that the test function expects
                return f"Signal{signal_idx + 1}.txt"

    def _run_test(self):
        """Run the selected test - results will print to terminal"""
        try:
            test_type = self.test_var.get()
            selected_signals = self.signal_listbox.curselection()

            if not selected_signals:
                messagebox.showwarning("Warning", "Please select a signal")
                return

            signal_idx = selected_signals[0]
            signal = self.gui.data_manager.get_signal(signal_idx)

            # Get expected file path
            expected_file = self.file_path_var.get()

            # Import test functions
            try:
                from Data.QuanTest1 import QuantizationTest1
                from Data.QuanTest2 import QuantizationTest2
                from Data.Task2Test import (
                    SignalSamplesAreEqual,
                    NormalizeSignal,
                    SubSignalSamplesAreEqual,
                    SinCosSignalSamplesAreEqual
                )
            except ImportError:
                messagebox.showerror("Error", "Test module 'Task2Test.py' not found")
                return

            # Run appropriate test - results will print to terminal
            if test_type == "squaring":
                result = self.gui.execute_operation("squaring", signal_index=signal_idx)
                if result:
                    result_signal = self.gui.data_manager.get_all_signals()[-1]
                    print(f"Running Squaring Test on signal: {signal['name']}")
                    SignalSamplesAreEqual(
                        "SQU",
                        expected_file,
                        result_signal["indices"],
                        result_signal["samples"]
                    )

            elif test_type == "accumulation":
                result = self.gui.execute_operation("accumulation", signal_index=signal_idx)
                if result:
                    result_signal = self.gui.data_manager.get_all_signals()[-1]
                    print(f"Running Accumulation Test on signal: {signal['name']}")
                    SignalSamplesAreEqual(
                        "ACC",
                        expected_file,
                        result_signal["indices"],
                        result_signal["samples"]
                    )

            elif test_type == "normalization":
                result = self.gui.execute_operation("normalization",
                                                    signal_index=signal_idx,
                                                    target_range=self.range_var.get())
                if result:
                    result_signal = self.gui.data_manager.get_all_signals()[-1]
                    print(f"Running Normalization Test on signal: {signal['name']}")
                    # For normalization, we need to use the NormalizeSignal function
                    # which expects MinRange and MaxRange parameters
                    min_range = -1 if self.range_var.get() == "-1_to_1" else 0
                    NormalizeSignal(
                        min_range,
                        1,
                        result_signal["indices"],
                        result_signal["samples"]
                    )

            elif test_type == "subtraction":
                selected_signals2 = self.subtraction_listbox.curselection()
                if not selected_signals2:
                    messagebox.showwarning("Warning", "Please select second signal")
                    return

                signal1_idx = selected_signals2[0]
                signal2 = self.gui.data_manager.get_signal(signal1_idx)

                # DEBUG: Print what signals we're working with
                print(f"Signal 1: {signal['name']}")
                print(f"Signal 2: {signal2['name']}")

                signal1_file = self._get_signal_file_name(signal_idx)
                signal2_file = self._get_signal_file_name(signal1_idx)

                print(f"Signal 1 file name for test: {signal1_file}")
                print(f"Signal 2 file name for test: {signal2_file}")

                result = self.gui.execute_operation("subtraction",
                                                    signal1_index=signal1_idx,
                                                    signal2_index=signal_idx)
                if result:
                    result_signal = self.gui.data_manager.get_all_signals()[-1]

                    print(f"Running Subtraction Test on signals: {signal['name']} - {signal2['name']}")

                    # FIX: Pass the actual file names that the test function expects
                    signal1_file = self._get_signal_file_name(signal_idx)
                    signal2_file = self._get_signal_file_name(signal1_idx)

                    print(f'running  {signal1_file} - {signal2_file}')
                    SubSignalSamplesAreEqual(
                        signal1_file,
                        signal2_file,
                        result_signal["indices"],
                        result_signal["samples"]
                    )

            elif test_type == "addition":
                selected_signals_add = self.addition_listbox.curselection()
                if len(selected_signals_add) < 1:
                    messagebox.showwarning("Warning", "Please select at least one additional signal to add")
                    return

                # Combine main signal with additional signals
                all_indices = [signal_idx] + list(selected_signals_add)
                result = self.gui.execute_operation("addition", signal_indices=all_indices)
                if result:
                    result_signal = self.gui.data_manager.get_all_signals()[-1]
                    print(f"Running Addition Test on {len(all_indices)} signals")
                    SignalSamplesAreEqual(
                        "Addition",
                        expected_file,
                        result_signal["indices"],
                        result_signal["samples"]
                    )

            elif test_type == "sinusoidal":
                # For sinusoidal test, we test an existing signal (should be a generated sine/cosine)
                print(f"Running Sinusoidal Test on signal: {signal['name']}")
                SinCosSignalSamplesAreEqual(
                    self.wave_type_var.get(),
                    expected_file,
                    signal["indices"],
                    signal["samples"]
                )

            elif test_type in ["quantization1", "quantization2"]:

                method = self.quant_method_var.get()
                try:
                    value = int(self.quant_value_var.get())
                    if value <= 0:
                        raise ValueError("Value must be positive")
                except ValueError:
                    messagebox.showerror("Error", "Please enter a positive integer for quantization value")
                    return

                from Data.signal_processor import SignalProcessor

                # Set parameters based on method
                if method == 'bits':
                    quantization_result = SignalProcessor.quantize_signal(signal, bits=value)
                else:
                    quantization_result = SignalProcessor.quantize_signal(signal, levels=value)

                # Extract the required data from signal processor for tests
                binary_encoded = quantization_result.get('binary_codes', [])
                quantized_samples = quantization_result['quantized_signal']['samples']
                error_samples = quantization_result['error_signal']['samples']
                interval_indices = quantization_result.get('interval_indices', [])

                # If interval_indices is not available, create 1-based indices
                if interval_indices is None or len(interval_indices) == 0:
                    interval_indices = [i + 1 for i in range(len(quantized_samples))]

                # Run the appropriate quantization test
                if not expected_file:
                    messagebox.showwarning("Warning", "Please select expected output file")
                    return

                if test_type == "quantization1":
                    QuantizationTest1(
                        expected_file,
                        binary_encoded,
                        quantized_samples
                    )
                elif test_type == "quantization2":
                    QuantizationTest2(
                        expected_file,
                        interval_indices,
                        binary_encoded,
                        quantized_samples,
                        error_samples
                    )

            # Close dialog after test
            self.dialog.destroy()
            messagebox.showinfo("Test Complete", "Test completed. Check terminal for results.")

        except Exception as e:
            messagebox.showerror("Error", f"Test failed:\n{str(e)}")
            import traceback
            traceback.print_exc()


# --------------------- task 3 ------------------
class FourierTestDialog:
    """Dialog for running Fourier Transform tests"""

    def __init__(self, parent, gui):
        self.parent = parent
        self.gui = gui
        self.dialog = None

    def show(self):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Fourier Transform Tests")
        self.dialog.geometry("500x400")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        tk.Label(self.dialog, text="Fourier Transform Test Suite",
                 font=("Arial", 12, "bold")).pack(pady=10)

        # Signal selection
        signal_frame = tk.Frame(self.dialog)
        signal_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(signal_frame, text="Select signal:").pack(anchor='w')
        self.signal_listbox = tk.Listbox(signal_frame, height=4, exportselection=False)
        self.signal_listbox.pack(fill=tk.X, pady=2)
        self._populate_signal_list()

        # Expected file selection
        file_frame = tk.Frame(self.dialog)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(file_frame, text="Expected output file:").pack(anchor='w')
        file_select_frame = tk.Frame(file_frame)
        file_select_frame.pack(fill=tk.X, pady=2)

        self.file_path_var = tk.StringVar()
        self.file_entry = tk.Entry(file_select_frame, textvariable=self.file_path_var,
                                   state='readonly', width=40)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(file_select_frame, text="Browse",
                  command=self._browse_file).pack(side=tk.RIGHT, padx=5)

        # Test buttons
        test_frame = tk.Frame(self.dialog)
        test_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tests = [
            ("DFT/IDFT Reconstruction Test", self._run_dft_idft_test),
            ("DC Removal Test", self._run_dc_removal_test),
            ("Compare with Expected Output", self._run_compare_test)
        ]

        for text, command in tests:
            tk.Button(
                test_frame,
                text=text,
                command=command,
                width=30
            ).pack(pady=5)

        tk.Button(
            test_frame,
            text="Close",
            command=self.dialog.destroy
        ).pack(pady=10)

    def _populate_signal_list(self):
        """Populate signal listbox"""
        self.signal_listbox.delete(0, tk.END)
        signals = self.gui.data_manager.get_all_signals()
        for i, signal in enumerate(signals):
            self.signal_listbox.insert(tk.END, f"{i}: {signal['name']}")

    def _browse_file(self):
        """Browse for expected output file"""
        file_path = filedialog.askopenfilename(
            title="Select Expected Output File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def _run_dft_idft_test(self):
        """Test DFT and IDFT reconstruction"""
        try:
            selected = self.signal_listbox.curselection()
            if not selected:
                messagebox.showwarning("Warning", "Please select a signal")
                return

            signal_idx = selected[0]
            signal = self.gui.data_manager.get_signal(signal_idx)

            # Ask for sampling frequency
            sampling_freq = simpledialog.askfloat(
                "Sampling Frequency",
                "Enter sampling frequency (Hz):",
                parent=self.dialog,
                minvalue=0.1,
                initialvalue=100.0
            )

            if sampling_freq is None:
                return

            # Compute DFT
            dft_result = SignalProcessor.compute_dft(signal, sampling_freq)

            # Compute IDFT
            reconstructed_signal = SignalProcessor.compute_idft(
                dft_result['components'], signal["indices"]
            )

            # Test reconstruction - USING EXISTING FUNCTION
            from Data.signalcompare import SignalComapreAmplitude
            success = SignalComapreAmplitude(signal["samples"], reconstructed_signal["samples"])

            if success:
                # Add reconstructed signal to manager
                self.gui.data_manager.add_signal(
                    reconstructed_signal,
                    f"Reconstructed_{signal['name']}"
                )
                self.gui.left_panel.update_signal_list()
                print("DFT/IDFT Test: PASSED")
            else:
                print("DFT/IDFT Test: FAILED")

        except Exception as e:
            messagebox.showerror("Error", f"DFT/IDFT test failed:\n{str(e)}")

    def _run_dc_removal_test(self):
        """Test DC removal"""
        try:
            selected = self.signal_listbox.curselection()
            if not selected:
                messagebox.showwarning("Warning", "Please select a signal")
                return

            signal_idx = selected[0]
            signal = self.gui.data_manager.get_signal(signal_idx)

            # Remove DC component - USE EXISTING METHOD OR CREATE ONE
            # First, you need to add this method to SignalProcessor:
            dc_removed_signal = SignalProcessor.remove_dc_component(signal)

            # Test DC removal - Check if mean is approximately zero
            mean_after_removal = np.mean(dc_removed_signal["samples"])
            success = abs(mean_after_removal) < 0.001  # Tolerance

            if success:
                # Add DC-removed signal to manager
                self.gui.data_manager.add_signal(
                    dc_removed_signal,
                    f"DC_Removed_{signal['name']}"
                )
                self.gui.left_panel.update_signal_list()
                print("DC Removal Test: PASSED")
            else:
                print(f"DC Removal Test: FAILED - Mean = {mean_after_removal}")

        except Exception as e:
            messagebox.showerror("Error", f"DC removal test failed:\n{str(e)}")

    def _run_compare_test(self):
        """Compare with expected output file"""
        try:
            selected = self.signal_listbox.curselection()
            expected_file = self.file_path_var.get()

            if not selected:
                messagebox.showwarning("Warning", "Please select a signal")
                return
            if not expected_file:
                messagebox.showwarning("Warning", "Please select expected output file")
                return

            signal_idx = selected[0]
            signal = self.gui.data_manager.get_signal(signal_idx)

            # Use the correct function from CompareSignals.py
            from Data.CompareSignals import SignalsAreEqual
            SignalsAreEqual(
                "Signal Comparison Test",  # TaskName parameter
                expected_file,
                signal["indices"],
                signal["samples"]
            )

        except Exception as e:
            messagebox.showerror("Error", f"Comparison test failed:\n{str(e)}")

# -----------------------------------------------------
def main():
    root = tk.Tk()
    app = SignalProcessingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
