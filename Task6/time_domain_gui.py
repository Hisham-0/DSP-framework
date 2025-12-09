"""
Time Domain Operations GUI - Task 6
Main GUI window with manual implementations of all 10 operations
"""
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class TimeDomainGUI:
    """Main GUI window for Time Domain Operations (Task 6)"""

    def __init__(self, root):
        self.root = root
        self.root.title("Time Domain Operations - Task 6")
        self.root.geometry("1200x700")

        # Store loaded signals
        self.signals = []  # List of dictionaries: {'name': str, 'indices': np.array, 'samples': np.array}
        self.display_modes = {}  # Signal index -> {'continuous': bool, 'discrete': bool}

        # Initialize state variables for plot
        self.panning = False
        self.pan_start = None
        self.zoom_mode_var = tk.StringVar(value="both")

        # Build the interface
        self._setup_menu()
        self._setup_main_frame()
        self._setup_left_panel()
        self._setup_plot_area()

    def _setup_menu(self):
        """Setup the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Signal", command=self.load_signal)
        file_menu.add_command(label="Save Signal", command=self.save_signal)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Operations menu
        ops_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Operations", menu=ops_menu)

        # Add all 10 operations
        ops_menu.add_command(label="1. Smoothing (Moving Average)",
                             command=self.smoothing_dialog)
        ops_menu.add_command(label="2. Sharpening (Derivative)",
                             command=self.sharpening_dialog)
        ops_menu.add_command(label="3. Delay/Advance",
                             command=self.shift_dialog)
        ops_menu.add_command(label="4. Fold (Time Reverse)",
                             command=self.fold_signal)
        ops_menu.add_command(label="5. Fold then Delay/Advance",
                             command=self.fold_and_shift_dialog)
        ops_menu.add_command(label="6. Remove DC Component",
                             command=self.remove_dc)
        ops_menu.add_command(label="7. Convolve Two Signals",
                             command=self.convolution_dialog)
        ops_menu.add_command(label="8. Normalized Cross/Auto-correlation",
                             command=self.correlation_dialog)
        ops_menu.add_command(label="9. Normalized Cross-correlation (Periodic)",
                             command=self.periodic_correlation_dialog)
        ops_menu.add_command(label="10. Time Delay Analysis (Periodic)",
                             command=self.time_delay_dialog)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_help)

    def _setup_main_frame(self):
        """Setup main container frame"""
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _setup_left_panel(self):
        """Setup left panel for signal list and controls"""
        # Left frame
        left_frame = tk.Frame(self.main_frame, width=300, bg='light gray',
                              relief=tk.RIDGE, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_frame.pack_propagate(False)

        # Title
        title_label = tk.Label(left_frame, text="Loaded Signals",
                               font=("Arial", 12, "bold"), bg='light gray')
        title_label.pack(pady=10)

        # Button frame
        btn_frame = tk.Frame(left_frame, bg='light gray')
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        load_btn = tk.Button(btn_frame, text="Load Signal",
                             command=self.load_signal, width=12)
        load_btn.pack(side=tk.LEFT, padx=2)

        delete_btn = tk.Button(btn_frame, text="Delete Selected",
                               command=self.delete_signal, width=12)
        delete_btn.pack(side=tk.LEFT, padx=2)

        # Treeview for signals
        tree_frame = tk.Frame(left_frame, bg='light gray')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create treeview
        columns = ('name', 'samples', 'continuous', 'discrete')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)

        # Define headings
        self.tree.heading('name', text='Signal Name')
        self.tree.heading('samples', text='Samples')
        self.tree.heading('continuous', text='C')
        self.tree.heading('discrete', text='D')

        # Configure columns
        self.tree.column('name', width=150, anchor='w')
        self.tree.column('samples', width=60, anchor='center')
        self.tree.column('continuous', width=30, anchor='center')
        self.tree.column('discrete', width=30, anchor='center')

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Pack tree and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind click events for checkboxes
        self.tree.bind('<Button-1>', self.on_tree_click)

        # Information label
        info_label = tk.Label(left_frame,
                              text="Click C/D to toggle display modes",
                              font=("Arial", 8), bg='light gray')
        info_label.pack(pady=5)

    def _setup_plot_area(self):
        """Setup main plot area"""
        # Right frame for plot
        right_frame = tk.Frame(self.main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Zoom controls
        zoom_frame = tk.Frame(right_frame)
        zoom_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        tk.Label(zoom_frame, text="Zoom Mode:").pack(side=tk.LEFT)

        tk.Radiobutton(zoom_frame, text="Both", variable=self.zoom_mode_var,
                       value="both").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(zoom_frame, text="X Only", variable=self.zoom_mode_var,
                       value="x_only").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(zoom_frame, text="Y Only", variable=self.zoom_mode_var,
                       value="y_only").pack(side=tk.LEFT, padx=5)

        # Plot buttons
        plot_btn_frame = tk.Frame(right_frame)
        plot_btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        tk.Button(plot_btn_frame, text="Reset View",
                  command=self.reset_plot).pack(side=tk.LEFT, padx=2)
        tk.Button(plot_btn_frame, text="Clear Plot",
                  command=self.clear_plot).pack(side=tk.LEFT, padx=2)

        # Matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Setup mouse interactions
        self._setup_mouse_interactions()

        # Setup initial empty plot
        self._setup_initial_plot()

    def _setup_mouse_interactions(self):
        """Connect mouse event handlers"""
        self.canvas.mpl_connect('scroll_event', self._on_mouse_wheel)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def _setup_initial_plot(self):
        """Setup initial empty plot"""
        self.ax.set_title("Time Domain Operations - No Signal Loaded", fontsize=12)
        self.ax.set_xlabel("Sample Index (n)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.fig.tight_layout()
        self.canvas.draw()

        # Save initial limits
        self.initial_xlim = self.ax.get_xlim()
        self.initial_ylim = self.ax.get_ylim()

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel zoom"""
        if event.inaxes != self.ax:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zoom_mode = self.zoom_mode_var.get()

        scale_factor = 0.8 if event.button == 'up' else 1.2
        xdata, ydata = event.xdata, event.ydata

        new_xlim, new_ylim = list(xlim), list(ylim)

        if zoom_mode in ["both", "x_only"] and xlim[1] > xlim[0]:
            new_width = (xlim[1] - xlim[0]) * scale_factor
            relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
            new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]

        if zoom_mode in ["both", "y_only"] and ylim[1] > ylim[0]:
            new_height = (ylim[1] - ylim[0]) * scale_factor
            rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
            new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def _on_mouse_press(self, event):
        """Handle mouse button press for panning"""
        if event.inaxes != self.ax or event.button != 1:
            return

        self.panning = True
        self.pan_start = (event.xdata, event.ydata)

    def _on_mouse_release(self, event):
        """Handle mouse button release"""
        if event.button == 1:
            self.panning = False
            self.pan_start = None

    def _on_mouse_move(self, event):
        """Handle mouse movement for panning"""
        if not self.panning or not self.pan_start or event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        dx = self.pan_start[0] - event.xdata
        dy = self.pan_start[1] - event.ydata

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self.canvas.draw_idle()

    def on_tree_click(self, event):
        """Handle clicks on treeview checkboxes"""
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        if item and column in ('#3', '#4'):  # Continuous or Discrete column
            idx = int(item)
            if idx < len(self.signals):
                if column == '#3':  # Continuous
                    current = self.display_modes.get(idx, {'continuous': False, 'discrete': False})
                    current['continuous'] = not current['continuous']
                    self.display_modes[idx] = current
                    char = '☑' if current['continuous'] else '☐'
                    self.tree.set(item, 'continuous', char)
                elif column == '#4':  # Discrete
                    current = self.display_modes.get(idx, {'continuous': False, 'discrete': False})
                    current['discrete'] = not current['discrete']
                    self.display_modes[idx] = current
                    char = '☑' if current['discrete'] else '☐'
                    self.tree.set(item, 'discrete', char)

                self.update_plot()

    def update_plot(self):
        """Update the plot with all signals"""
        self.ax.clear()

        if not self.signals:
            self._setup_initial_plot()
            return

        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

        for idx, signal in enumerate(self.signals):
            display = self.display_modes.get(idx, {'continuous': False, 'discrete': False})
            color = colors[idx % len(colors)]

            indices = signal['indices']
            samples = signal['samples']
            label = signal['name']

            if display['continuous']:
                self.ax.plot(indices, samples, color=color, linewidth=2,
                             alpha=0.8, label=f"{label} (Continuous)")

            if display['discrete']:
                markerline, stemlines, baseline = self.ax.stem(
                    indices, samples, linefmt=f'{color}-', markerfmt=f'{color}o',
                    basefmt=" ", label=f"{label} (Discrete)"
                )
                plt.setp(markerline, markersize=4, color=color)
                plt.setp(stemlines, linewidth=1, color=color, alpha=0.7)

        self.ax.set_title("Time Domain Signals", fontsize=12)
        self.ax.set_xlabel("Sample Index (n)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.7)

        if self.signals:
            self.ax.legend(loc='upper right', fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw()

    def reset_plot(self):
        """Reset plot view to initial zoom"""
        if self.ax.has_data():
            self.ax.set_xlim(self.initial_xlim)
            self.ax.set_ylim(self.initial_ylim)
            self.canvas.draw_idle()

    def clear_plot(self):
        """Clear the plot"""
        self.ax.clear()
        self._setup_initial_plot()

    def load_signal(self):
        """Load signal from file in the specified format"""
        file_path = filedialog.askopenfilename(
            title="Select Signal File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    lines = f.read().strip().split('\n')

                if len(lines) < 3:
                    raise ValueError("Invalid file format: not enough lines")

                # Parse header
                signal_type = int(lines[0].strip())  # 0=Time, 1=Frequency
                is_periodic = int(lines[1].strip())  # 0 or 1
                n_samples = int(lines[2].strip())  # Number of samples/frequencies

                if len(lines) != 3 + n_samples:
                    raise ValueError(f"Expected {n_samples} data rows, got {len(lines) - 3}")

                indices = []
                samples = []

                if signal_type == 0:  # Time Domain
                    # Format: [Index SampleAmp]
                    for i in range(n_samples):
                        parts = lines[3 + i].strip().split()
                        if len(parts) < 2:
                            continue
                        indices.append(float(parts[0]))
                        samples.append(float(parts[1]))

                else:  # Frequency Domain (SignalType=1)
                    # Format: [Freq Amp PhaseShift]
                    # You need to decide how to handle frequency domain signals
                    # For now, we'll skip them or convert them
                    messagebox.showwarning("Warning",
                                           "Frequency domain signals not supported for time domain operations.\n"
                                           "Only time domain signals (SignalType=0) can be processed.")
                    return

                # Convert to numpy arrays
                indices = np.array(indices)
                samples = np.array(samples)

                # Create signal dictionary
                signal = {
                    'name': f"{os.path.basename(file_path)}",
                    'signal_type': signal_type,
                    'is_periodic': bool(is_periodic),
                    'indices': indices,
                    'samples': samples,
                    'file_path': file_path
                }

                self.signals.append(signal)
                self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

                # Update treeview
                self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                                 values=(signal['name'], len(samples), '☑', '☐'))

                # Update plot
                self.update_plot()

                messagebox.showinfo("Success",
                                    f"Loaded signal:\n"
                                    f"Type: {'Time' if signal_type == 0 else 'Frequency'} Domain\n"
                                    f"Periodic: {'Yes' if is_periodic else 'No'}\n"
                                    f"Samples: {len(samples)}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load signal: {str(e)}")

    def save_signal(self):
        """Save signal to file in the specified format"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a signal to save")
            return

        idx = int(selected[0])
        if idx >= len(self.signals):
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Signal",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                signal = self.signals[idx]
                with open(file_path, 'w') as f:
                    # Write header
                    f.write(f"{signal.get('signal_type', 0)}\n")
                    f.write(f"{1 if signal.get('is_periodic', False) else 0}\n")
                    f.write(f"{len(signal['samples'])}\n")

                    # Write data
                    for i in range(len(signal['samples'])):
                        if signal.get('signal_type', 0) == 0:  # Time domain
                            index = signal['indices'][i] if i < len(signal['indices']) else i
                            f.write(f"{index} {signal['samples'][i]}\n")
                        else:  # Frequency domain (unlikely in this app)
                            # Handle frequency format if needed
                            pass

                messagebox.showinfo("Success", f"Signal saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save signal: {str(e)}")

    def delete_signal(self):
        """Delete selected signal"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a signal to delete")
            return

        idx = int(selected[0])
        if idx < len(self.signals):
            # Remove signal
            del self.signals[idx]

            # Update display modes
            new_display_modes = {}
            for i in range(len(self.signals)):
                if i >= idx:
                    new_display_modes[i] = self.display_modes.get(i + 1, {'continuous': False, 'discrete': False})
                else:
                    new_display_modes[i] = self.display_modes.get(i, {'continuous': False, 'discrete': False})
            self.display_modes = new_display_modes

            # Clear and rebuild tree
            for item in self.tree.get_children():
                self.tree.delete(item)

            for i, signal in enumerate(self.signals):
                display = self.display_modes.get(i, {'continuous': False, 'discrete': False})
                cont_char = '☑' if display['continuous'] else '☐'
                disc_char = '☑' if display['discrete'] else '☐'
                self.tree.insert('', 'end', iid=str(i),
                                 values=(signal['name'], len(signal['samples']), cont_char, disc_char))

            # Update plot
            self.update_plot()

    def get_selected_signal(self):
        """Get the currently selected signal"""
        selected = self.tree.selection()
        if not selected:
            return None
        idx = int(selected[0])
        if idx < len(self.signals):
            return idx, self.signals[idx]
        return None

    def get_two_selected_signals(self):
        """Get two selected signals (for operations requiring two inputs)"""
        selected = self.tree.selection()
        if len(selected) != 2:
            return None

        idx1 = int(selected[0])
        idx2 = int(selected[1])

        if idx1 < len(self.signals) and idx2 < len(self.signals):
            return (idx1, self.signals[idx1]), (idx2, self.signals[idx2])
        return None

    # Operation dialog methods
    def smoothing_dialog(self):
        """Show smoothing dialog"""
        from time_domain_operations import TimeDomainOperations

        signal_info = self.get_selected_signal()
        if not signal_info:
            messagebox.showwarning("Warning", "Please select a signal")
            return

        idx, signal = signal_info

        # Ask for window size
        window = simpledialog.askinteger("Smoothing",
                                         "Enter window size (odd number):",
                                         parent=self.root,
                                         minvalue=1,
                                         initialvalue=3)
        if window is None:
            return

        if window % 2 == 0:
            messagebox.showwarning("Warning", "Window size should be odd for symmetric smoothing")
            window += 1

        try:
            # Apply smoothing
            y = TimeDomainOperations.smoothing_moving_average(signal['samples'], window)

            # Create new signal
            new_signal = {
                'name': f"Smoothed_{signal['name']}_w{window}",
                'indices': signal['indices'][:len(y)],
                'samples': y
            }

            # Add to signals list
            self.signals.append(new_signal)
            self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

            # Update treeview
            self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                             values=(new_signal['name'], len(y), '☑', '☐'))

            # Update plot
            self.update_plot()

            messagebox.showinfo("Success", f"Applied smoothing with window size {window}")

        except Exception as e:
            messagebox.showerror("Error", f"Smoothing failed: {str(e)}")

    def sharpening_dialog(self):
        """Show sharpening dialog"""
        from time_domain_operations import TimeDomainOperations

        signal_info = self.get_selected_signal()
        if not signal_info:
            messagebox.showwarning("Warning", "Please select a signal")
            return

        idx, signal = signal_info

        # Ask for derivative type
        choice = simpledialog.askstring("Sharpening",
                                        "Enter '1' for First derivative or '2' for Second derivative:",
                                        parent=self.root)
        if choice is None:
            return

        try:
            if choice.strip() == '1':
                y = TimeDomainOperations.first_derivative(signal['samples'])
                title = "First_Derivative"
            else:
                y = TimeDomainOperations.second_derivative(signal['samples'])
                title = "Second_Derivative"

            # Create new signal
            new_signal = {
                'name': f"{title}_{signal['name']}",
                'indices': signal['indices'][:len(y)],
                'samples': y
            }

            # Add to signals list
            self.signals.append(new_signal)
            self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

            # Update treeview
            self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                             values=(new_signal['name'], len(y), '☑', '☐'))

            # Update plot
            self.update_plot()

            messagebox.showinfo("Success", "Applied sharpening operation")

        except Exception as e:
            messagebox.showerror("Error", f"Sharpening failed: {str(e)}")

    def shift_dialog(self):
        """Show shift dialog"""
        from time_domain_operations import TimeDomainOperations

        signal_info = self.get_selected_signal()
        if not signal_info:
            messagebox.showwarning("Warning", "Please select a signal")
            return

        idx, signal = signal_info

        # Ask for shift amount
        k = simpledialog.askinteger("Shift",
                                    "Enter k (positive for delay, negative for advance):",
                                    parent=self.root)
        if k is None:
            return

        try:
            y = TimeDomainOperations.shift_signal(signal['samples'], k)

            # Create new signal
            direction = "Delay" if k > 0 else "Advance"
            new_signal = {
                'name': f"{direction}_{abs(k)}_samples_{signal['name']}",
                'indices': signal['indices'],
                'samples': y
            }

            # Add to signals list
            self.signals.append(new_signal)
            self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

            # Update treeview
            self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                             values=(new_signal['name'], len(y), '☑', '☐'))

            # Update plot
            self.update_plot()

            messagebox.showinfo("Success", f"Shifted signal by {k} samples")

        except Exception as e:
            messagebox.showerror("Error", f"Shift operation failed: {str(e)}")

    def fold_signal(self):
        """Fold (time reverse) selected signal"""
        from time_domain_operations import TimeDomainOperations

        signal_info = self.get_selected_signal()
        if not signal_info:
            messagebox.showwarning("Warning", "Please select a signal")
            return

        idx, signal = signal_info

        try:
            y = TimeDomainOperations.fold_signal(signal['samples'])

            # Create new signal
            new_signal = {
                'name': f"Folded_{signal['name']}",
                'indices': signal['indices'],
                'samples': y
            }

            # Add to signals list
            self.signals.append(new_signal)
            self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

            # Update treeview
            self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                             values=(new_signal['name'], len(y), '☑', '☐'))

            # Update plot
            self.update_plot()

            messagebox.showinfo("Success", "Applied folding operation")

        except Exception as e:
            messagebox.showerror("Error", f"Folding failed: {str(e)}")

    def fold_and_shift_dialog(self):
        """Fold and then shift signal"""
        from time_domain_operations import TimeDomainOperations

        signal_info = self.get_selected_signal()
        if not signal_info:
            messagebox.showwarning("Warning", "Please select a signal")
            return

        idx, signal = signal_info

        # Ask for shift amount
        k = simpledialog.askinteger("Shift after folding",
                                    "Enter k (positive for delay, negative for advance):",
                                    parent=self.root)
        if k is None:
            return

        try:
            # First fold, then shift
            folded = TimeDomainOperations.fold_signal(signal['samples'])
            y = TimeDomainOperations.shift_signal(folded, k)

            # Create new signal
            direction = "Delay" if k > 0 else "Advance"
            new_signal = {
                'name': f"Folded_{direction}_{abs(k)}_samples_{signal['name']}",
                'indices': signal['indices'] + k,
                'samples': y
            }

            # Add to signals list
            self.signals.append(new_signal)
            self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

            # Update treeview
            self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                             values=(new_signal['name'], len(y), '☑', '☐'))

            # Update plot
            self.update_plot()

            messagebox.showinfo("Success", f"Applied fold and shift by {k} samples")

        except Exception as e:
            messagebox.showerror("Error", f"Fold and shift operation failed: {str(e)}")

    def remove_dc(self):
        """Remove DC component from selected signal"""
        from time_domain_operations import TimeDomainOperations

        signal_info = self.get_selected_signal()
        if not signal_info:
            messagebox.showwarning("Warning", "Please select a signal")
            return

        idx, signal = signal_info

        try:
            y = TimeDomainOperations.remove_dc(signal['samples'])

            # Create new signal
            new_signal = {
                'name': f"DC_Removed_{signal['name']}",
                'indices': signal['indices'],
                'samples': y
            }

            # Add to signals list
            self.signals.append(new_signal)
            self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

            # Update treeview
            self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                             values=(new_signal['name'], len(y), '☑', '☐'))

            # Update plot
            self.update_plot()

            messagebox.showinfo("Success", "Removed DC component")

        except Exception as e:
            messagebox.showerror("Error", f"DC removal failed: {str(e)}")

    def convolution_dialog(self):
        """Show convolution dialog"""
        from time_domain_operations import TimeDomainOperations

        signals = self.get_two_selected_signals()
        if not signals:
            messagebox.showwarning("Warning", "Please select exactly two signals")
            return

        (idx1, signal1), (idx2, signal2) = signals

        try:
            y = TimeDomainOperations.convolve_signals(signal1['samples'], signal2['samples'])

            # Create new signal
            indices = np.arange(len(y))
            new_signal = {
                'name': f"Convolution_{signal1['name']}_{signal2['name']}",
                'indices': indices,
                'samples': y
            }

            # Add to signals list
            self.signals.append(new_signal)
            self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

            # Update treeview
            self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                             values=(new_signal['name'], len(y), '☑', '☐'))

            # Update plot
            self.update_plot()

            messagebox.showinfo("Success", "Applied convolution operation")

        except Exception as e:
            messagebox.showerror("Error", f"Convolution failed: {str(e)}")

    def correlation_dialog(self):
        """Show correlation dialog"""
        from time_domain_operations import TimeDomainOperations

        # Ask for correlation type
        choice = simpledialog.askstring("Correlation",
                                        "Enter 'auto' for autocorrelation or 'cross' for cross-correlation:",
                                        parent=self.root)
        if choice is None:
            return

        choice = choice.strip().lower()

        try:
            if choice == 'auto':
                signal_info = self.get_selected_signal()
                if not signal_info:
                    messagebox.showwarning("Warning", "Please select a signal")
                    return

                idx, signal = signal_info
                corr = TimeDomainOperations.normalized_autocorrelation(signal['samples'])
                indices = np.arange(-(len(signal['samples']) - 1), len(signal['samples']))
                name = f"AutoCorrelation_{signal['name']}"

            else:  # cross-correlation
                signals = self.get_two_selected_signals()
                if not signals:
                    messagebox.showwarning("Warning", "Please select exactly two signals")
                    return

                (idx1, signal1), (idx2, signal2) = signals
                corr = TimeDomainOperations.normalized_cross_correlation(signal1['samples'], signal2['samples'])
                indices = np.arange(-(len(signal2['samples']) - 1), len(signal1['samples']))
                name = f"CrossCorrelation_{signal1['name']}_{signal2['name']}"

            # Create new signal
            new_signal = {
                'name': name,
                'indices': indices,
                'samples': corr
            }

            # Add to signals list
            self.signals.append(new_signal)
            self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

            # Update treeview
            self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                             values=(new_signal['name'], len(corr), '☑', '☐'))

            # Update plot
            self.update_plot()

            messagebox.showinfo("Success", "Applied correlation operation")

        except Exception as e:
            messagebox.showerror("Error", f"Correlation failed: {str(e)}")

    def periodic_correlation_dialog(self):
        """Show periodic correlation dialog"""
        from time_domain_operations import TimeDomainOperations

        signals = self.get_two_selected_signals()
        if not signals:
            messagebox.showwarning("Warning", "Please select exactly two signals")
            return

        (idx1, signal1), (idx2, signal2) = signals

        try:
            corr = TimeDomainOperations.circular_cross_correlation_periodic(signal1['samples'], signal2['samples'])

            # Create new signal
            indices = np.arange(len(corr))
            new_signal = {
                'name': f"Periodic_CrossCorrelation_{signal1['name']}_{signal2['name']}",
                'indices': indices,
                'samples': corr
            }

            # Add to signals list
            self.signals.append(new_signal)
            self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

            # Update treeview
            self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                             values=(new_signal['name'], len(corr), '☑', '☐'))

            # Update plot
            self.update_plot()

            messagebox.showinfo("Success", "Applied periodic cross-correlation")

        except Exception as e:
            messagebox.showerror("Error", f"Periodic correlation failed: {str(e)}")

    def time_delay_dialog(self):
        """Show time delay analysis dialog"""
        from time_domain_operations import TimeDomainOperations

        signals = self.get_two_selected_signals()
        if not signals:
            messagebox.showwarning("Warning", "Please select exactly two signals")
            return

        (idx1, signal1), (idx2, signal2) = signals

        # Ask for sampling period
        ts = simpledialog.askfloat("Sampling Period",
                                   "Enter sampling period (seconds):",
                                   parent=self.root,
                                   minvalue=0.0001,
                                   initialvalue=0.001)
        if ts is None:
            return

        try:
            signed_lag, time_delay, corr = TimeDomainOperations.time_delay_analysis_periodic(
                signal1['samples'], signal2['samples'], ts)

            # Show results in a message box
            result_text = f"Time Delay Analysis Results:\n"
            result_text += f"Signed Lag (samples): {signed_lag}\n"
            result_text += f"Time Delay (seconds): {time_delay:.6f}\n"
            result_text += f"Maximum Correlation: {corr[signed_lag % len(corr)]:.6f}"

            # Create new signal for correlation
            indices = np.arange(len(corr))
            new_signal = {
                'name': f"TimeDelay_Correlation_{signal1['name']}_{signal2['name']}",
                'indices': indices,
                'samples': corr
            }

            # Add to signals list
            self.signals.append(new_signal)
            self.display_modes[len(self.signals) - 1] = {'continuous': True, 'discrete': False}

            # Update treeview
            self.tree.insert('', 'end', iid=str(len(self.signals) - 1),
                             values=(new_signal['name'], len(corr), '☑', '☐'))

            # Update plot
            self.update_plot()

            messagebox.showinfo("Time Delay Analysis", result_text)

        except Exception as e:
            messagebox.showerror("Error", f"Time delay analysis failed: {str(e)}")

    def show_help(self):
        """Show help information"""
        help_text = """Time Domain Operations - Task 6

All 10 operations implemented with manual calculations:

1. Smoothing: Moving average with specified window size
2. Sharpening: First and second derivatives
3. Delay/Advance: Shift signal by k samples
4. Folding: Time reversal of signal
5. Fold & Shift: Combined folding and shifting
6. Remove DC: Subtract mean value
7. Convolution: Linear convolution of two signals
8. Correlation: Normalized cross/auto-correlation
9. Periodic Correlation: Circular correlation for periodic signals
10. Time Delay: Estimate delay between periodic signals

Instructions:
1. Load signals using 'Load Signal' button
2. Select signal(s) from the list
3. Choose operation from menu
4. Results are displayed and added to signal list
5. Toggle continuous/discrete display by clicking C/D columns

Note: All operations use manual implementations without built-in numpy functions."""

        messagebox.showinfo("Help - Time Domain Operations", help_text)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = TimeDomainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()