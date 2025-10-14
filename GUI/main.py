"""
Signal Processing Framework - Main Entry Point
==================================================
This is the main application file that initializes and runs the signal processing GUI.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import os
import sys

# Import the left panel and plot area components
from left_panel import LeftPanel
from plot_area import PlotArea

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

        # Display Menu
        self._create_display_menu(menubar)

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
        tk.Label(self.dialog, text="Signal 1 (Minuend):").pack(pady=5)
        self.signal1_list = tk.Listbox(self.dialog, height=4, exportselection=False)
        self.signal1_list.pack(padx=10, pady=2, fill=tk.X)

        tk.Label(self.dialog, text="Signal 2 (Subtrahend):").pack(pady=5)
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
                    signal1_index=sel1[0],
                    signal2_index=sel2[0]
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


class SignalProcessingGUI:
    """Main GUI Controller Class"""

    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processing Framework")
        self.root.state('zoomed')

        self.data_manager = SignalDataManager()
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
                self.data_manager.add_signal(signal_data, signal_name)
                self.left_panel.update_signal_list()
                messagebox.showinfo(
                    "Signal Loaded",
                    f"Signal: {signal_name}\n"
                    f"Samples: {signal_data['num_samples']}"
                )
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

#-----------------------------------------------------------
# Operations call from the logic layer
#-----------------------------------------------------------

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


def main():
    root = tk.Tk()
    app = SignalProcessingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()