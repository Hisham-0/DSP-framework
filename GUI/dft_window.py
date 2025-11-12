import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os
from Data.signal_processor import SignalProcessor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class DFTWindow:
    """Modal window for Fourier Transform analysis and manipulation"""

    def __init__(self, parent, gui, signal, sampling_freq):
        self.parent = parent
        self.gui = gui
        self.signal = signal
        self.sampling_freq = sampling_freq
        SignalProcessor_ins = SignalProcessor()

        # Compute DFT - handle the dictionary response
        self.dft_result = SignalProcessor_ins.compute_dft(SignalProcessor_ins, signal["samples"])
        self.num_samples = len(signal["samples"])

        # Store original components for reset functionality
        self.original_components = self.dft_result.copy()

        # Normalization factors - computed from actual amplitudes
        self.amp_min = 0
        self.amp_max = 1

        self._create_window()
        self._create_ui()
        self.refresh_all()

    def _create_window(self):
        """Create the modal window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Frequency Domain: {self.signal['name']}")
        self.window.geometry("1200x750")
        self.window.transient(self.parent)
        self.window.grab_set()

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_ui(self):
        """Create the UI layout"""
        # Main paned window
        main_pane = ttk.PanedWindow(self.window, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Top info frame
        info_frame = ttk.Frame(main_pane)
        main_pane.add(info_frame, weight=0)
        self._create_info_panel(info_frame)

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

    def _create_info_panel(self, parent):
        """Create info panel showing normalization info"""
        frame = ttk.LabelFrame(parent, text="Normalization Info")
        frame.pack(fill=tk.X, padx=5, pady=5)

        self.info_label = ttk.Label(frame, text="", font=("Arial", 9))
        self.info_label.pack(pady=5, padx=10)

    def _create_amplitude_plot(self, parent):
        """Create amplitude spectrum plot"""
        plot_frame = ttk.Frame(parent)
        parent.add(plot_frame, weight=1)

        ttk.Label(plot_frame, text="Amplitude Spectrum (Normalized)",
                  font=("Arial", 10, "bold")).pack(pady=2)

        self.fig_amp, self.ax_amp = plt.subplots(figsize=(6, 4))
        self.canvas_amp = FigureCanvasTkAgg(self.fig_amp, plot_frame)
        self.canvas_amp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_phase_plot(self, parent):
        """Create phase spectrum plot"""
        plot_frame = ttk.Frame(parent)
        parent.add(plot_frame, weight=1)

        ttk.Label(plot_frame, text="Phase Spectrum",
                  font=("Arial", 10, "bold")).pack(pady=2)

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

        btn_frame1 = ttk.Frame(control_frame)
        btn_frame1.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame1, text="Remove DC Component",
                   command=self.remove_dc).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        ttk.Button(btn_frame1, text="Reconstruct Signal",
                   command=self.reconstruct_signal).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        ttk.Button(btn_frame1, text="Reset All",
                   command=self.reset_components).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        # Instructions
        inst_frame = ttk.LabelFrame(parent, text="Instructions")
        inst_frame.pack(fill=tk.X, padx=5, pady=5)


    def _create_component_table(self, parent):
        """Create table to display frequency components"""
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(table_frame, text="Frequency Components",
                  font=("Arial", 10, "bold")).pack(pady=2)

        columns = ("index", "freq", "amp", "phase", "dominant")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)

        # Configure columns
        self.tree.heading("index", text="Index")
        self.tree.heading("freq", text="Frequency (Hz)")
        self.tree.heading("amp", text="Norm. Amplitude")
        self.tree.heading("phase", text="Phase (rad)")
        self.tree.heading("dominant", text="Dominant")

        self.tree.column("index", width=60, anchor=tk.CENTER)
        self.tree.column("freq", width=120, anchor=tk.CENTER)
        self.tree.column("amp", width=120, anchor=tk.CENTER)
        self.tree.column("phase", width=120, anchor=tk.CENTER)
        self.tree.column("dominant", width=80, anchor=tk.CENTER)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind double-click for editing
        self.tree.bind("<Double-1>", self.on_component_edit)

    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Components", command=self.export_components)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.on_close)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Reset to Original", command=self.reset_components)
        edit_menu.add_command(label="Remove DC Component", command=self.remove_dc)

    def _normalize_amplitudes(self, actual_amplitudes):
        """Normalize amplitudes to 0-1 range"""
        if len(actual_amplitudes) == 0:
            return actual_amplitudes

        self.amp_min = np.min(actual_amplitudes)
        self.amp_max = np.max(actual_amplitudes)

        if self.amp_max == self.amp_min:
            return np.ones_like(actual_amplitudes) * 0.5

        # Normalize to [0, 1]
        normalized = (actual_amplitudes - self.amp_min) / (self.amp_max - self.amp_min)
        return normalized

    def _denormalize_amplitude(self, normalized_value):
        """
        Convert normalized amplitude back to actual amplitude
        Handles values outside [0, 1] range:
        - norm_value = 1.0 → amp_max
        - norm_value = 0.0 → amp_min
        - norm_value > 1.0 → scales proportionally beyond max
        - norm_value < 0.0 → scales proportionally below min
        """
        range_val = self.amp_max - self.amp_min

        if normalized_value >= 1.0:
            # Scale beyond max: 2.0 → 2 * amp_max, etc.
            return self.amp_min + normalized_value * range_val
        elif normalized_value <= 0.0:
            # Scale below min: -1.0 → -range, etc.
            return self.amp_min + normalized_value * range_val
        else:
            # Normal range [0, 1]
            return self.amp_min + normalized_value * range_val

    def refresh_all(self):
        """Update plots, table, and info panel"""
        # Get frequency domain data
        frequencies, actual_amplitudes, phases = self.gui.signal_processor.get_dft_plot_data(
            self.dft_result, self.sampling_freq
        )

        # Normalize amplitudes
        normalized_amplitudes = self._normalize_amplitudes(actual_amplitudes)

        # Update info label
        self.info_label.config(
            text=f"Amplitude Range: Min = {self.amp_min:.4f}, Max = {self.amp_max:.4f} | "
                 f"Normalized: 0 = Min, 1 = Max | Sampling Freq: {self.sampling_freq} Hz"
        )

        # Update amplitude plot
        self.ax_amp.clear()
        self.ax_amp.stem(frequencies, normalized_amplitudes, linefmt='C0-',
                         markerfmt='C0o', basefmt='C0:')
        self.ax_amp.set_title("Frequency vs Normalized Amplitude", fontsize=12, fontweight='bold')
        self.ax_amp.set_xlabel("Frequency (Hz)", fontsize=10)
        self.ax_amp.set_ylabel("Normalized Amplitude", fontsize=10)
        self.ax_amp.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Dominant Threshold')
        self.ax_amp.axhline(y=1.0, color='g', linestyle='--', alpha=0.3, label='Max')
        self.ax_amp.axhline(y=0.0, color='b', linestyle='--', alpha=0.3, label='Min')

        # Adjust y-axis to show extended range if needed
        y_min = min(-0.1, np.min(normalized_amplitudes) - 0.1)
        y_max = max(1.1, np.max(normalized_amplitudes) + 0.1)
        self.ax_amp.set_ylim(y_min, y_max)
        self.ax_amp.grid(True, alpha=0.3)
        self.ax_amp.legend(loc='upper right', fontsize=8)

        # Update phase plot
        self.ax_phase.clear()
        self.ax_phase.stem(frequencies, phases, linefmt='C1-',
                           markerfmt='C1o', basefmt='C1:')
        self.ax_phase.set_title("Frequency vs Phase", fontsize=12, fontweight='bold')
        self.ax_phase.set_xlabel("Frequency (Hz)", fontsize=10)
        self.ax_phase.set_ylabel("Phase (radians)", fontsize=10)
        self.ax_phase.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_phase.grid(True, alpha=0.3)

        self.canvas_amp.draw()
        self.canvas_phase.draw()

        # Update table
        self._update_component_table(frequencies, normalized_amplitudes, phases)

    def _update_component_table(self, frequencies, normalized_amplitudes, phases):
        """Update the component table with color coding for dominant frequencies"""
        self.tree.delete(*self.tree.get_children())

        for i in range(len(frequencies)):
            # Mark dominant components (normalized amplitude ≥ 0.5)
            is_dominant = normalized_amplitudes[i] >= 0.5
            dominant_text = "✓" if is_dominant else ""

            item = self.tree.insert("", "end", values=(
                f"{i}",
                f"{frequencies[i]:.3f}",
                f"{normalized_amplitudes[i]:.4f}",
                f"{phases[i]:.4f}",
                dominant_text
            ))

            # Color code dominant components
            if is_dominant:
                self.tree.item(item, tags=('dominant',))

        # Configure tag colors
        self.tree.tag_configure('dominant', background='#lightgreen')

    def on_component_edit(self, event):
        """Handle component editing with separate amplitude and phase editing"""
        selection = self.tree.selection()
        if not selection:
            return

        item = selection[0]
        values = self.tree.item(item)['values']
        index = int(values[0])

        # Get current values
        current_actual_amp = np.abs(self.dft_result[index])
        current_phase = np.angle(self.dft_result[index])

        # Get current normalized amplitude
        frequencies, actual_amps, _ = self.gui.signal_processor.get_dft_plot_data(
            self.dft_result, self.sampling_freq
        )
        normalized_amps = self._normalize_amplitudes(actual_amps)
        current_norm_amp = normalized_amps[index]

        # Create edit dialog
        self._show_edit_dialog(index, current_norm_amp, current_phase)

    def _show_edit_dialog(self, index, current_norm_amp, current_phase):
        """Show dialog for editing component amplitude and phase"""
        edit_window = tk.Toplevel(self.window)
        edit_window.title(f"Edit Component {index}")
        edit_window.geometry("450x400")
        edit_window.transient(self.window)
        edit_window.grab_set()

        # Main frame
        main_frame = ttk.Frame(edit_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame,
                                text=f"Edit Frequency Component {index}",
                                font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 10))

        # Amplitude section
        amp_frame = ttk.LabelFrame(main_frame, text="Normalized Amplitude", padding="10")
        amp_frame.pack(fill=tk.X, pady=5)

        ttk.Label(amp_frame,
                  text="Enter normalized value (0=min, 1=max):",
                  font=("Arial", 9)).pack(anchor=tk.W)

        amp_var = tk.DoubleVar(value=round(current_norm_amp, 4))
        amp_entry = ttk.Entry(amp_frame, textvariable=amp_var, width=20, font=("Arial", 10))
        amp_entry.pack(pady=5)

        ttk.Label(amp_frame,
                  text="Tip: Values >1 scale beyond max, <0 scale below min",
                  font=("Arial", 8), foreground="gray").pack(anchor=tk.W)

        # Phase section
        phase_frame = ttk.LabelFrame(main_frame, text="Phase", padding="10")
        phase_frame.pack(fill=tk.X, pady=5)

        ttk.Label(phase_frame,
                  text="Enter phase in radians:",
                  font=("Arial", 9)).pack(anchor=tk.W)

        phase_var = tk.DoubleVar(value=round(current_phase, 4))
        phase_entry = ttk.Entry(phase_frame, textvariable=phase_var, width=20, font=("Arial", 10))
        phase_entry.pack(pady=5)

        ttk.Label(phase_frame,
                  text=f"Tip: π = {np.pi:.4f}, -π = {-np.pi:.4f}",
                  font=("Arial", 8), foreground="gray").pack(anchor=tk.W)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        def apply_changes():
            try:
                new_norm_amp = float(amp_var.get())
                new_phase = float(phase_var.get())

                # Convert normalized amplitude to actual amplitude
                new_actual_amp = self._denormalize_amplitude(new_norm_amp)

                # Update the DFT component
                self.dft_result[index] = new_actual_amp * np.exp(1j * new_phase)

                edit_window.destroy()
                self.refresh_all()

                messagebox.showinfo("Success",
                                    f"Component {index} updated successfully!\n"
                                    f"Normalized Amp: {new_norm_amp:.4f}\n"
                                    f"Phase: {new_phase:.4f} rad")

            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values")

        def cancel():
            edit_window.destroy()

        ttk.Button(button_frame, text="Apply Changes",
                   command=apply_changes).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(button_frame, text="Cancel",
                   command=cancel).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Focus on amplitude entry
        amp_entry.focus()
        amp_entry.select_range(0, tk.END)

    def remove_dc(self):
        """Remove DC component (F(0))"""
        response = messagebox.askyesno(
            "Remove DC Component",
            "Remove the DC component (frequency = 0)?\n"
            "This will set F(0) to zero."
        )
        if response:
            self.dft_result[0] = 0 + 0j
            self.refresh_all()
            messagebox.showinfo("Success", "DC component removed")

    def reset_components(self):
        """Reset all components to their original values"""
        response = messagebox.askyesno(
            "Reset Components",
            "Reset all frequency components to their original values?"
        )
        if response:
            self.dft_result = self.original_components.copy()
            self.refresh_all()
            messagebox.showinfo("Success", "Components reset to original values")

    def reconstruct_signal(self):
        """Reconstruct signal using IDFT"""
        try:
            # Compute IDFT
            time, samples = SignalProcessor.compute_idft(
                self.dft_components, self.num_samples, self.signal["indices"]
            )

            # Create new signal dictionary
            reconstructed_signal = {
                "indices": time,
                "samples": samples,
                "signal_type": 0,  # Continuous signal
                "is_periodic": self.signal["is_periodic"],
                "num_samples": len(samples)
            }

            # Add to data manager
            signal_name = f"Reconstructed_{self.signal['name']}"
            self.gui.data_manager.add_signal(reconstructed_signal, signal_name)

            # Update UI
            self.gui.left_panel.update_signal_list()
            self.gui.update_plot_display()

            messagebox.showinfo("Success",
                                f"Signal reconstructed successfully!\n"
                                f"Name: {signal_name}\n"
                                f"Samples: {len(samples)}")

        except Exception as e:
            messagebox.showerror("Error", f"Reconstruction failed:\n{str(e)}")

    def export_components(self):
        """Export frequency components to file"""
        try:
            from tkinter import filedialog

            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Frequency Components"
            )

            if filename:
                frequencies, actual_amps, phases = self.gui.signal_processor.get_dft_plot_data(
                    self.dft_components, self.sampling_freq
                )
                normalized_amps = self._normalize_amplitudes(actual_amps)

                with open(filename, 'w') as f:
                    for i in range(len(frequencies)):
                        dominant = "YES" if normalized_amps[i] >= 0.5 else "NO"
                        f.write(f"{i:<8}{frequencies[i]:<15.4f}{normalized_amps[i]:<15.6f}"
                                f"{phases[i]:<15.6f}{dominant:<10}\n")

                messagebox.showinfo("Success", f"Components exported to:\n{filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")

    def on_close(self):
        """Close the window"""
        self.window.grab_release()
        self.window.destroy()
