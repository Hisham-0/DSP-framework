"""
Main Plotting Area with Interactive Controls
"""
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt


class PlotArea:
    """Main Plotting Area with Interactive Controls"""

    def __init__(self, parent, gui):
        self.parent = parent
        self.gui = gui

        # Initialize state variables
        self.panning = False
        self.pan_start = None
        self.zoom_mode_var = tk.StringVar(value="both")
        self.initial_xlim = (None, None)
        self.initial_ylim = (None, None)

        self._create_plot_area()

    def _create_plot_area(self):
        """Create the main plot area with controls"""
        # Main container frame
        self.main_frame = tk.Frame(self.parent)
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Zoom controls at the top
        self._create_zoom_controls()

        # Matplotlib plot area
        self._create_matplotlib_area()

    def _create_zoom_controls(self):
        """Create zoom mode radio buttons"""
        zoom_frame = tk.Frame(self.main_frame)
        zoom_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        tk.Label(
            zoom_frame,
            text="Zoom Mode:",
            font=("Arial", 8)
        ).pack(side=tk.LEFT)

        # Zoom mode radio buttons
        tk.Radiobutton(
            zoom_frame,
            text="Both",
            variable=self.zoom_mode_var,
            value="both",
            font=("Arial", 8)
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            zoom_frame,
            text="X Only",
            variable=self.zoom_mode_var,
            value="x_only",
            font=("Arial", 8)
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            zoom_frame,
            text="Y Only",
            variable=self.zoom_mode_var,
            value="y_only",
            font=("Arial", 8)
        ).pack(side=tk.LEFT, padx=5)

    def _create_matplotlib_area(self):
        """Create and configure the matplotlib figure and canvas"""
        # Create figure and axis
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Create canvas widget
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Setup mouse event handlers
        self._setup_mouse_interactions()

        # Setup initial empty plot
        self._setup_initial_plot()

    def _setup_mouse_interactions(self):
        """Connect mouse event handlers to the canvas"""
        self.canvas.mpl_connect('scroll_event', self._on_mouse_wheel)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def _setup_initial_plot(self):
        """Setup the initial empty plot with labels and grid"""
        self.ax.set_title("Signal Display Area - No Signal Loaded", fontsize=12)
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.fig.tight_layout()
        self.canvas.draw()

        # Save initial limits for reset functionality
        self.initial_xlim = self.ax.get_xlim()
        self.initial_ylim = self.ax.get_ylim()

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel zoom events"""
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
            new_xlim = [
                xdata - new_width * (1 - relx),
                xdata + new_width * relx
            ]

        if zoom_mode in ["both", "y_only"] and ylim[1] > ylim[0]:
            new_height = (ylim[1] - ylim[0]) * scale_factor
            rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
            new_ylim = [
                ydata - new_height * (1 - rely),
                ydata + new_height * rely
            ]

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
        """Handle mouse button release to stop panning"""
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

    def plot_signals(self, signals_data):
        """Plot multiple signals with their specified display modes"""
        self.ax.clear()

        if not signals_data:
            self._setup_initial_plot()
            return

        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        color_idx = 0

        for signal_idx, display_info in signals_data.items():
            signal = self.gui.data_manager.get_signal(signal_idx)
            if not signal:
                continue

            color = colors[color_idx % len(colors)]
            color_idx += 1

            indices = np.array(signal["indices"])
            samples = np.array(signal["samples"])
            label = signal['name']

            if display_info['continuous']:
                self.ax.plot(
                    indices, samples,
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                    label=f"{label} (Continuous)"
                )

            if display_info['discrete']:
                markerline, stemlines, baseline = self.ax.stem(
                    indices, samples,
                    linefmt=f'{color}-',
                    markerfmt=f'{color}o',
                    basefmt=" ",
                    label=f"{label} (Discrete)"
                )
                plt.setp(markerline, markersize=4, color=color)
                plt.setp(stemlines, linewidth=1, color=color, alpha=0.7)

        self.ax.set_title("Signal Display", fontsize=12)
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.7)

        if signals_data:
            self.ax.legend(loc='best', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

        self.initial_xlim = self.ax.get_xlim()
        self.initial_ylim = self.ax.get_ylim()

    def reset_plot_view(self):
        """Reset the plot view to initial zoom and pan state"""
        if not self.ax.has_data():
            return

        self.ax.set_xlim(self.initial_xlim)
        self.ax.set_ylim(self.initial_ylim)
        self.canvas.draw_idle()

    def clear_plot(self):
        """Clear the plot and restore initial empty state"""
        self.ax.clear()
        self._setup_initial_plot()