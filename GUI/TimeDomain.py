# time_domain.py
"""
Time Domain menu module for Tkinter app.

Usage:
    from time_domain import attach_time_domain_menu
    attach_time_domain_menu(root, menubar)

Dependencies:
    numpy, matplotlib
"""

import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from Data.signal_processor import SignalDataManager
from GUI.left_panel import LeftPanel


##########################
# Utility / I/O functions
##########################

def parse_signal_string(s):
    """Parse a string of numbers separated by comma/space/newline into numpy array."""
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return np.array([])
    # accept commas, spaces, semicolons
    s = s.replace(";", ",").replace("\n", ",")
    parts = [p for p in (item.strip() for item in s.replace(",", " ").split()) if p != ""]
    try:
        arr = np.array([float(p) for p in parts], dtype=float)
    except Exception as e:
        raise ValueError(f"Couldn't parse signal: {e}")
    return arr


def load_signal_from_file(parent):
    """Open file dialog to load numbers (comma separated or one per line)."""
    path = filedialog.askopenfilename(parent=parent, title="Open signal file",
                                      filetypes=[("Text files", "*.txt;*.csv"), ("All files", "*.*")])
    if not path:
        return None
    with open(path, "r") as f:
        text = f.read()
    return parse_signal_string(text)


def save_signal_to_file(parent, signal):
    """Ask user for folder/name and save the signal as a .txt file."""
    path = filedialog.asksaveasfilename(
        parent=parent,
        defaultextension=".txt",
        filetypes=[("Text File", "*.txt"), ("All files", "*.*")]
    )
    if not path:
        return
    try:
        with open(path, "w") as f:
            for value in signal:
                f.write(f"{value}\n")
        messagebox.showinfo("Saved", f"Signal saved successfully:\n{path}", parent=parent)
    except Exception as e:
        messagebox.showerror("Error", f"Could not save file:\n{e}", parent=parent)


def ensure_nonempty(arr, name="signal"):
    if arr is None or arr.size == 0:
        raise ValueError(f"{name} is empty")


def show_plot(title, t, y, y2=None, legend=None, text_output=None):
    """Create a Toplevel with a matplotlib plot, textual output, and Save button."""
    win = tk.Toplevel()
    win.title(title)
    
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(t, y)
    if y2 is not None:
        ax.plot(t, y2, linestyle="--")
    if legend:
        ax.legend(legend)
    ax.grid(True)
    ax.set_xlabel("n (sample index)")
    ax.set_ylabel("Amplitude")

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()

    # Show text output
    if text_output is not None:
        txt = tk.Text(win, height=6)
        txt.pack(fill="both", expand=False)
        txt.insert("end", text_output)
        txt.configure(state="disabled")

    # Save button
    def save():
        save_signal_to_file(win, y)

    save_button = tk.Button(win, text="Save Signal as .txt", command=save)
    save_button.pack(pady=5)


##########################
# DSP operations
##########################

def smoothing_moving_average(x, window):
    """Simple moving average: y[n] = mean of window points centered/causal? 
    We'll implement a simple causal moving average: average of current and previous (window-1) samples."""
    if window < 1:
        raise ValueError("Window must be at least 1")
    x = np.asarray(x, dtype=float)
    y = np.convolve(x, np.ones(window)/window, mode='full')[:x.size]  # causal-like (starts from first sample)
    return y


def first_derivative(x):
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    # y[n] = x[n] - x[n-1], for n=0 we set x[-1]=0 (or same as x[0]); we choose zero padding at left
    y[0] = x[0]  # x0 - 0
    y[1:] = x[1:] - x[:-1]
    return y


def second_derivative(x):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    # using y[n] = x[n+1] - 2x[n] + x[n-1]
    N = x.size
    if N == 0:
        return y
    # handle boundaries with zero-padding
    padded = np.pad(x, (1, 1), 'constant', constant_values=(0.0, 0.0))
    for n in range(N):
        y[n] = padded[n+2] - 2*padded[n+1] + padded[n]
    return y


def shift_signal(x, k):
    """Shift by k steps. positive k -> delay (right shift), negative k -> advance (left shift).
    We'll fill introduced samples with zeros."""
    x = np.asarray(x, dtype=float)
    N = x.size
    if k == 0:
        return x.copy()
    y = np.zeros(N, dtype=float)
    if k > 0:
        if k >= N:
            return y
        y[k:] = x[:N-k]
    else:  # k < 0
        kabs = abs(k)
        if kabs >= N:
            return y
        y[:N-kabs] = x[kabs:]
    return y


def fold_signal(x):
    """Folding (time reversal): y[n] = x[-n]. We'll return reversed array.
    For discrete sequence x[0..N-1], folding is x_rev[n] = x[N-1-n].
    """
    x = np.asarray(x, dtype=float)
    return x[::-1].copy()


def remove_dc(x):
    x = np.asarray(x, dtype=float)
    return x - np.mean(x) if x.size else x


def convolve_signals(x, h):
    return np.convolve(x, h, mode='full')


def normalized_cross_correlation(x, y):
    """Compute normalized cross-correlation (linear) using numpy.correlate, normalized so that
    at zero-lag of identical signals equals 1."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("Empty input signal")
    corr = np.correlate(x - x.mean(), y - y.mean(), mode='full')
    denom = np.sqrt(np.sum((x - x.mean())**2) * np.sum((y - y.mean())**2))
    if denom == 0:
        return corr * 0.0
    return corr / denom


def normalized_autocorrelation(x):
    return normalized_cross_correlation(x, x)


def circular_cross_correlation_periodic(x, y):
    """Compute normalized cross-correlation of periodic signals (circular). Works for unequal lengths by
    computing circular correlation over L = lcm(len(x), len(y)) where each sequence is tiled to L."""
    lx = len(x)
    ly = len(y)
    if lx == 0 or ly == 0:
        raise ValueError("Empty input")
    L = np.lcm(lx, ly)
    X = np.tile(x, int(L // lx))
    Y = np.tile(y, int(L // ly))
    # circular cross-correlation can be computed by FFT conj(X)*Y etc, but we'll use np.roll correlation:
    # compute normalized linear cross-correlation and then map to circular by sampling only lags [0..L-1]
    # simpler: compute via FFT:
    fx = np.fft.fft(X)
    fy = np.fft.fft(Y)
    corr = np.fft.ifft(np.conj(fx) * fy)
    corr = np.real(corr)
    # normalize
    denom = math.sqrt(np.sum((X - X.mean())**2) * np.sum((Y - Y.mean())**2))
    if denom == 0:
        return corr * 0.0
    return corr / denom


def time_delay_analysis_periodic(x, y, ts):
    """Given two periodic signals x and y and sampling period ts, estimate the time delay of y relative to x.
    We'll compute circular cross-correlation and find the lag at maximum correlation."""
    ensure_nonempty(x, "signal x")
    ensure_nonempty(y, "signal y")
    corr = circular_cross_correlation_periodic(x, y)
    # find lag of max
    lag_index = int(np.argmax(corr))
    L = corr.size
    # Map lag index to signed lag in samples: if lag_index > L/2, interpret as negative lag offset (wrap-around)
    if lag_index > L // 2:
        signed_lag = lag_index - L
    else:
        signed_lag = lag_index
    time_delay = signed_lag * ts
    return signed_lag, time_delay, corr


##########################
# GUI callbacks
##########################

def ask_for_signal(parent, title="Enter signal", prompt="Enter samples (comma or space separated)."):
    resp = simpledialog.askstring(title=title, prompt=prompt, parent=parent)
    if resp is None:
        return None
    try:
        return parse_signal_string(resp)
    except Exception as e:
        messagebox.showerror("Parse error", f"Couldn't parse signal: {e}", parent=parent)
        return None
    

def require_one_or_two_signals(parent, allow_file=True, two=False):
    """Helper to ask user for signal(s) either by typing or loading file."""
    sig1 = None
    if messagebox.askyesno("Load from file?", "Load first signal from file? (No=type/paste)", parent=parent):
        sig1 = load_signal_from_file(parent)
    else:
        sig1 = ask_for_signal(parent, title="Signal 1", prompt="Enter signal 1 samples (comma or space separated):")
    if sig1 is None:
        return None if not two else (None, None)
    if not two:
        return sig1
    # second signal
    sig2 = None
    if messagebox.askyesno("Load from file?", "Load second signal from file? (No=type/paste)", parent=parent):
        sig2 = load_signal_from_file(parent)
    else:
        sig2 = ask_for_signal(parent, title="Signal 2", prompt="Enter signal 2 samples (comma or space separated):")
    return sig1, sig2


def action_smoothing(parent):
    x = require_one_or_two_signals(parent, two=False)
    if x is None:
        return
    w = simpledialog.askinteger("Window", "Number of points to average (integer >=1):", parent=parent, minvalue=1)
    if w is None:
        return
    try:
        y = smoothing_moving_average(x, w)
    except Exception as e:
        messagebox.showerror("Error", str(e), parent=parent)
        return
    t = np.arange(y.size)
    show_plot(f"Smoothing (window={w})", t, y, text_output=f"y = {np.array2string(y, precision=5, separator=',')}")


def action_sharpening(parent):
    x = require_one_or_two_signals(parent, two=False)
    if x is None:
        return
    choice = simpledialog.askstring("Sharpening",
                                    "Enter 1 for First derivative, 2 for Second derivative:",
                                    parent=parent)
    if choice is None:
        return
    if choice.strip() == "1":
        y = first_derivative(x)
        title = "First Derivative y[n] = x[n] - x[n-1]"
    else:
        y = second_derivative(x)
        title = "Second Derivative y[n] = x[n+1] - 2x[n] + x[n-1]"
    t = np.arange(y.size)
    show_plot(title, t, y, text_output=f"y = {np.array2string(y, precision=5, separator=',')}")


def action_shift(parent):
    x = require_one_or_two_signals(parent, two=False)
    if x is None:
        return
    k = simpledialog.askinteger("Shift", "Enter k (positive delay, negative advance):", parent=parent)
    if k is None:
        return
    y = shift_signal(x, k)
    t = np.arange(y.size)
    show_plot(f"Shifted by {k} samples", t, y,
              text_output=f"y = {np.array2string(y, precision=5, separator=',')}")


def action_fold(parent):
    x = require_one_or_two_signals(parent, two=False)
    if x is None:
        return
    y = fold_signal(x)
    t = np.arange(y.size)
    show_plot("Folded (time-reversed) signal", t, y, text_output=f"y = {np.array2string(y, precision=5, separator=',')}")


def action_fold_and_shift(parent):
    x = require_one_or_two_signals(parent, two=False)
    if x is None:
        return
    y_folded = fold_signal(x)
    k = simpledialog.askinteger("Shift folded", "Enter k (positive delay, negative advance):", parent=parent)
    if k is None:
        return
    y = shift_signal(y_folded, k)
    t = np.arange(y.size)
    show_plot(f"Folded and shifted by {k}", t, y,
              text_output=f"y = {np.array2string(y, precision=5, separator=',')}")


def action_remove_dc(parent):
    x = require_one_or_two_signals(parent, two=False)
    if x is None:
        return
    y = remove_dc(x)
    t = np.arange(y.size)
    show_plot("DC removed (x - mean(x))", t, y, text_output=f"y = {np.array2string(y, precision=5, separator=',')}")


def action_convolution(parent):
    # two signals required
    x, h = require_one_or_two_signals(parent, two=True)
    if x is None or h is None:
        return
    y = convolve_signals(x, h)
    t = np.arange(y.size)
    show_plot("Convolution (x * h)", t, y,
              legend=["y = conv(x,h)"],
              text_output=f"y = {np.array2string(y, precision=5, separator=',')}")


def action_cross_or_auto_correlation(parent):
    mode = simpledialog.askstring("Correlation",
                                  "Enter 'auto' for autocorrelation or 'cross' for cross-correlation:",
                                  parent=parent)
    if mode is None:
        return
    if mode.strip().lower() == "auto":
        x = require_one_or_two_signals(parent, two=False)
        if x is None:
            return
        corr = normalized_autocorrelation(x)
        t = np.arange(- (len(x) - 1), len(x))
        # np.correlate gives full length 2N-1; our normalized_autocorrelation uses same shape
        show_plot("Normalized Autocorrelation", t, corr,
                  text_output=f"r = {np.array2string(corr, precision=5, separator=',')}")
    else:
        x, y = require_one_or_two_signals(parent, two=True)
        if x is None or y is None:
            return
        corr = normalized_cross_correlation(x, y)
        # lags from -(len(y)-1) to (len(x)-1) (depending on which is which); compute lags length
        L = corr.size
        # center lags around zero for display
        lag_start = - (len(y) - 1)
        t = np.arange(lag_start, lag_start + L)
        show_plot("Normalized Cross-correlation", t, corr,
                  text_output=f"r = {np.array2string(corr, precision=5, separator=',')}")


def action_periodic_cross_correlation(parent):
    x, y = require_one_or_two_signals(parent, two=True)
    if x is None or y is None:
        return
    corr = circular_cross_correlation_periodic(x, y)
    L = corr.size
    t = np.arange(0, L)  # circular lags 0..L-1
    show_plot("Normalized Circular Cross-correlation (Periodic)", t, corr,
              text_output=f"r_circ = {np.array2string(corr, precision=5, separator=',')}")


def action_time_delay_analysis(parent):
    x, y = require_one_or_two_signals(parent, two=True)
    if x is None or y is None:
        return
    ts = simpledialog.askfloat("Sampling period",
                               "Enter sampling period (seconds) (e.g., 0.001):",
                               parent=parent, minvalue=1e-12)
    if ts is None:
        return
    try:
        signed_lag, time_delay, corr = time_delay_analysis_periodic(x, y, ts)
    except Exception as e:
        messagebox.showerror("Error", str(e), parent=parent)
        return
    L = corr.size
    t = np.arange(0, L)
    text = f"Estimated signed lag (samples): {signed_lag}\nEstimated time delay (seconds): {time_delay}\n"
    text += f"Correlation peak index: {int(np.argmax(corr))}\n"
    show_plot("Time Delay Analysis (Periodic signals)", t, corr, text_output=text)


##########################
# Attach to menubar
##########################

def attach_time_domain_menu(root, menubar):
    """
    Attach a 'Time Domain' menu to an existing menubar (tk.Menu).

    Example:
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        attach_time_domain_menu(root, menubar)
    """
    time_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Time Domain", menu=time_menu)

    time_menu.add_command(label="Smoothing (Moving Average)", command=lambda: action_smoothing(root))
    time_menu.add_command(label="Sharpening (First/Second Derivative)", command=lambda: action_sharpening(root))
    time_menu.add_command(label="Delay/Advance by k", command=lambda: action_shift(root))
    time_menu.add_command(label="Fold (Time Reverse)", command=lambda: action_fold(root))
    time_menu.add_command(label="Fold then Delay/Advance", command=lambda: action_fold_and_shift(root))
    time_menu.add_command(label="Remove DC Component", command=lambda: action_remove_dc(root))
    time_menu.add_command(label="Convolve two signals", command=lambda: action_convolution(root))
    time_menu.add_command(label="Normalized Cross/Auto-correlation", command=lambda: action_cross_or_auto_correlation(root))
    time_menu.add_command(label="Normalized Cross-correlation (Periodic)", command=lambda: action_periodic_cross_correlation(root))
    time_menu.add_command(label="Time Delay Analysis (Periodic signals)", command=lambda: action_time_delay_analysis(root))

    # optional separator and help
    time_menu.add_separator()
    time_menu.add_command(label="Help / Usage", command=lambda: messagebox.showinfo(
        "Time Domain Help",
        "Use menu items to enter or load signals. Signals are comma/space-separated numbers.\n"
        "Smoothing: causal moving average.\n"
        "Sharpening: first/second discrete derivatives.\n"
        "Delay/Advance: positive k delays (shifts right), negative k advances (shifts left).\n"
        "Folding: time reversal.\n"
        "Convolution: full linear convolution (length = Nx+Nh-1).\n"
        "Cross/Auto-correlation: normalized (so identical signals produce peak 1).\n"
        "Periodic cross-correlation: circular correlation using tiling to LCM length.\n"
        "Time delay analysis: finds peak of circular cross-correlation and returns sample lag * sampling period.",
        parent=root))


# If run standalone, create a test window
# if __name__ == "__main__":
#     root = tk.Tk()
#     root.title("Time Domain Module Test")
#     menubar = tk.Menu(root)
#     root.config(menu=menubar)
#     attach_time_domain_menu(root, menubar)
#     tk.Label(root, text="Time Domain test menu added. Use the menu to try features.").pack(padx=10, pady=10)
#     root.mainloop()