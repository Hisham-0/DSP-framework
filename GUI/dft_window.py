import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Data.signal_processor import SignalProcessor

class DFTWindow:
    """Modal window for Fourier Transform analysis and manipulation"""

    def __init__(self, parent, gui, signal, sampling_freq):
        self.parent = parent
        self.gui = gui
        self.signal = signal
        self.sampling_freq = sampling_freq

        # Compute DFT - handle the dictionary response
        dft_result = self.gui.signal_processor.compute_dft(signal, sampling_freq)
        self.dft_components = dft_result['components']  # Extract the components array
        self.num_samples = len(signal["samples"])

        self._create_window()
        self._create_ui()
        self.refresh_all()

    def _create_window(self):
        """Create the modal window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Frequency Domain: {self.signal['name']}")
        self.window.geometry("1000x700")
        self.window.transient(self.parent)
        self.window.grab_set()

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_ui(self):
        """Create the UI layout"""
        # Main paned window
        main_pane = ttk.PanedWindow(self.window, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Plots pane
        plots_pane = ttk.PanedWindow(main_pane, orient=tk.HORIZONTAL)
        main_pane.add(plots_pane, weight=2)

        # Amplitude plot
        self._create_amplitude_plot(plots_pane)

        # Phase plot
        self._create_phase_plot(plots_pane)

        # Controls pane
        controls_pane = ttk.Frame(main_pane)
        main_pane.add(controls_pane, weight=1)

        self._create_controls(controls_pane)
        self._create_menu()

    def _create_amplitude_plot(self, parent):
        """Create amplitude spectrum plot"""
        plot_frame = ttk.Frame(parent)
        parent.add(plot_frame, weight=1)

        self.fig_amp, self.ax_amp = plt.subplots(figsize=(6, 4))
        self.canvas_amp = FigureCanvasTkAgg(self.fig_amp, plot_frame)
        self.canvas_amp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_phase_plot(self, parent):
        """Create phase spectrum plot"""
        plot_frame = ttk.Frame(parent)
        parent.add(plot_frame, weight=1)

        self.fig_phase, self.ax_phase = plt.subplots(figsize=(6, 4))
        self.canvas_phase = FigureCanvasTkAgg(self.fig_phase, plot_frame)
        self.canvas_phase.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_controls(self, parent):
        """Create control buttons and table"""
        # Table for components
        self._create_component_table(parent)

        # Control buttons
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="Remove DC Component",
                   command=self.remove_dc).pack(fill=tk.X, pady=2)

        ttk.Button(control_frame, text="Reconstruct Signal",
                   command=self.reconstruct_signal).pack(fill=tk.X, pady=2)

        # Component modification frame
        mod_frame = ttk.LabelFrame(parent, text="Modify Selected Components")
        mod_frame.pack(fill=tk.X, padx=5, pady=5)

        self._create_modification_controls(mod_frame)

    def _create_component_table(self, parent):
        """Create table to display frequency components"""
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("index", "freq", "amp", "phase", "dominant")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)

        # Configure columns
        self.tree.heading("index", text="Index")
        self.tree.heading("freq", text="Freq (Hz)")
        self.tree.heading("amp", text="Amplitude")
        self.tree.heading("phase", text="Phase (rad)")
        self.tree.heading("dominant", text="Dominant")

        self.tree.column("index", width=60)
        self.tree.column("freq", width=100)
        self.tree.column("amp", width=100)
        self.tree.column("phase", width=100)
        self.tree.column("dominant", width=80)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind double-click for editing
        self.tree.bind("<Double-1>", self.on_component_edit)

    def _create_modification_controls(self, parent):
        """Create controls for modifying components"""
        # Amplitude modification
        amp_frame = ttk.Frame(parent)
        amp_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(amp_frame, text="Amplitude:").pack(side=tk.LEFT)
        self.amp_var = tk.DoubleVar(value=0.1)
        ttk.Entry(amp_frame, textvariable=self.amp_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(amp_frame, text="Add",
                   command=lambda: self.modify_selected("amp_add")).pack(side=tk.LEFT, padx=2)
        ttk.Button(amp_frame, text="Multiply",
                   command=lambda: self.modify_selected("amp_mult")).pack(side=tk.LEFT, padx=2)

        # Phase modification
        phase_frame = ttk.Frame(parent)
        phase_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(phase_frame, text="Phase:").pack(side=tk.LEFT)
        self.phase_var = tk.DoubleVar(value=0.1)
        ttk.Entry(phase_frame, textvariable=self.phase_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(phase_frame, text="Add",
                   command=lambda: self.modify_selected("phase_add")).pack(side=tk.LEFT, padx=2)
        ttk.Button(phase_frame, text="Multiply",
                   command=lambda: self.modify_selected("phase_mult")).pack(side=tk.LEFT, padx=2)

    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Close", command=self.on_close)

    def refresh_all(self):
        """Update plots and table"""
        frequencies, normalized_amplitudes, phases = self.gui.signal_processor.get_dft_plot_data(
            self.dft_components, self.sampling_freq
        )

        # Update amplitude plot
        self.ax_amp.clear()
        self.ax_amp.stem(frequencies, normalized_amplitudes, linefmt='C0-', markerfmt='C0o')
        self.ax_amp.set_title("Frequency vs Amplitude")
        self.ax_amp.set_xlabel("Frequency (Hz)")
        self.ax_amp.set_ylabel("Normalized Amplitude")
        self.ax_amp.grid(True)

        # Update phase plot
        self.ax_phase.clear()
        self.ax_phase.stem(frequencies, phases, linefmt='C1-', markerfmt='C1o')
        self.ax_phase.set_title("Frequency vs Phase")
        self.ax_phase.set_xlabel("Frequency (Hz)")
        self.ax_phase.set_ylabel("Phase (radians)")
        self.ax_phase.grid(True)

        self.canvas_amp.draw()
        self.canvas_phase.draw()

        # Update table
        self._update_component_table(frequencies, normalized_amplitudes, phases)

    def _update_component_table(self, frequencies, amplitudes, phases):
        """Update the component table"""
        self.tree.delete(*self.tree.get_children())

        for i in range(len(frequencies)):
            dominant = "YES" if amplitudes[i] > 0.5 else ""

            self.tree.insert("", "end", values=(
                f"{i}",
                f"{frequencies[i]:.3f}",
                f"{amplitudes[i]:.4f}",
                f"{phases[i]:.4f}",
                dominant
            ))

    def on_component_edit(self, event):
        """Handle component editing"""
        item = self.tree.selection()[0]
        values = self.tree.item(item)['values']
        index = int(values[0])

        # Simple editing - you can enhance this like in your dft_window.py
        new_amp = simpledialog.askfloat("Edit Amplitude", "Enter new amplitude:",
                                        initialvalue=float(values[2]))
        if new_amp is not None:
            self.modify_component_amplitude(index, new_amp)

    def modify_component_amplitude(self, index, new_amp):
        """Modify a specific component's amplitude"""
        # Get current component
        current = self.dft_components[index]
        current_amp = np.abs(current)
        current_phase = np.angle(current)

        # Update with new amplitude
        self.dft_components[index] = new_amp * np.exp(1j * current_phase)
        self.refresh_all()

    def modify_selected(self, operation):
        """Modify selected components"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Select components first")
            return

        for item in selected:
            index = int(self.tree.item(item)['values'][0])
            self._apply_modification(index, operation)

        self.refresh_all()

    def _apply_modification(self, index, operation):
        """Apply specific modification to a component"""
        current = self.dft_components[index]
        current_amp = np.abs(current)
        current_phase = np.angle(current)

        if operation == "amp_add":
            new_amp = current_amp + self.amp_var.get()
        elif operation == "amp_mult":
            new_amp = current_amp * self.amp_var.get()
        elif operation == "phase_add":
            new_phase = current_phase + self.phase_var.get()
        elif operation == "phase_mult":
            new_phase = current_phase * self.phase_var.get()
        else:
            return

        if "amp" in operation:
            self.dft_components[index] = new_amp * np.exp(1j * current_phase)
        else:
            self.dft_components[index] = current_amp * np.exp(1j * new_phase)

    def remove_dc(self):
        """Remove DC component (F(0))"""
        self.dft_components[0] = 0 + 0j
        self.refresh_all()

    def reconstruct_signal(self):
        """Reconstruct signal using IDFT"""
        try:
            time, samples = SignalProcessor.compute_idft(
                self.dft_components, self.num_samples, self.signal["indices"]
            )

            # Create new signal
            reconstructed_signal = {
                "indices": time,
                "samples": samples,
                "signal_type": 0,
                "is_periodic": self.signal["is_periodic"],
                "num_samples": len(samples)
            }

            # Add to data manager
            self.gui.data_manager.add_signal(
                reconstructed_signal,
                f"Reconstructed_{self.signal['name']}"
            )

            # Update UI
            self.gui.left_panel.update_signal_list()
            self.gui.update_plot_display()

            messagebox.showinfo("Success", "Signal reconstructed successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Reconstruction failed: {str(e)}")

    def on_close(self):
        """Close the window"""
        self.window.grab_release()
        self.window.destroy()