"""
Left panel with signal list and operations
"""
import tkinter as tk
from tkinter import ttk, messagebox

class LeftPanel:
    """Left panel containing signal list and operations"""

    def __init__(self, parent, gui):
        self.parent = parent
        self.gui = gui
        self.signal_vars = {}  # Store checkbutton variables for each signal

        self.create_panel()

    def create_panel(self):
        """Create the left panel"""
        # Main frame with gray background - smaller width
        self.main_frame = tk.Frame(self.parent, bg='light gray', relief=tk.RIDGE, borderwidth=1, width=350)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=2, pady=2)
        self.main_frame.pack_propagate(False)

        # Signal Management Section
        self.create_signal_management_section()

        # Signal List Section - Compact
        self.create_signal_list_section()

        # Operations Section
        self.create_operations_section()

        # Generation Section
        self.create_generation_section()

    def create_signal_management_section(self):
        """Create signal management buttons"""
        management_frame = tk.Frame(self.main_frame, bg='light gray')
        management_frame.pack(fill=tk.X, padx=5, pady=3)

        # Load button
        load_btn = tk.Button(management_frame, text="Load Signal",
                           command=self.gui.load_signal,
                           width=10, font=("Arial", 8))
        load_btn.pack(side=tk.LEFT, padx=2)

        # Delete button
        delete_btn = tk.Button(management_frame, text="Delete",
                             command=self.gui.delete_signal,
                             width=10, font=("Arial", 8))
        delete_btn.pack(side=tk.LEFT, padx=2)

    def create_signal_list_section(self):
        """Create compact signal list with checkboxes"""
        # Scalable parameters
        TITLE_FONT_SIZE = 9
        TITLE_PADY = 2
        TREE_HEIGHT = 8
        TREE_PADX = 5
        TREE_PADY = 2
        TREE_IPADY = 2
        NAME_COLUMN_WIDTH = 180
        CHECKBOX_COLUMN_WIDTH = 20

        # Title
        list_label = tk.Label(self.main_frame, text="Loaded Signals",
                              font=("Arial", TITLE_FONT_SIZE, "bold"), bg='light gray')
        list_label.pack(pady=TITLE_PADY)

        # Create treeview for signals with checkboxes - very compact
        tree_frame = tk.Frame(self.main_frame, bg='light gray')
        tree_frame.pack(fill=tk.BOTH, expand=False, padx=TREE_PADX, pady=TREE_PADY, ipady=TREE_IPADY)

        # Create treeview - scalable height
        columns = ('name', 'continuous', 'discrete')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=TREE_HEIGHT)

        # Define headings
        self.tree.heading('name', text='Name')
        self.tree.heading('continuous', text='C')
        self.tree.heading('discrete', text='D')

        # Configure columns - scalable widths
        self.tree.column('name', width=NAME_COLUMN_WIDTH, anchor='w')
        self.tree.column('continuous', width=CHECKBOX_COLUMN_WIDTH, anchor='center')
        self.tree.column('discrete', width=CHECKBOX_COLUMN_WIDTH, anchor='center')

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=v_scrollbar.set)

        # Pack treeview and scrollbars
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind click events for checkboxes
        self.tree.bind('<Button-1>', self.on_tree_click)

    def create_operations_section(self):
        """Create operations buttons section"""
        # Scalable parameters
        TITLE_FONT_SIZE = 9
        TITLE_PADY = 2
        FRAME_PADX = 5
        FRAME_PADY = 2
        BUTTON_WIDTH = 8
        BUTTON_FONT_SIZE = 8
        BUTTON_PADX = 1
        BUTTON_PADY = 1

        # Title
        ops_label = tk.Label(self.main_frame, text="Operations",
                             font=("Arial", TITLE_FONT_SIZE, "bold"), bg='light gray')
        ops_label.pack(pady=TITLE_PADY)

        # Operations frame
        ops_frame = tk.Frame(self.main_frame, bg='light gray')
        ops_frame.pack(fill=tk.X, padx=FRAME_PADX, pady=FRAME_PADY)

        # Operation buttons (2 columns) - scalable buttons
        operations = [
            ("Add", self.gui.show_addition_dialog),
            ("Subtract", self.gui.show_subtraction_dialog),
            ("Multiply", self.gui.show_multiplication_dialog),
            ("Square", self.gui.show_squaring_dialog),
            ("Normalize", self.gui.show_normalization_dialog),
            ("Accumulate", self.gui.show_accumulation_dialog),
        ]

        for i, (text, command) in enumerate(operations):
            row = i // 2
            col = i % 2
            btn = tk.Button(ops_frame, text=text, command=command,
                            width=BUTTON_WIDTH, font=("Arial", BUTTON_FONT_SIZE))
            btn.grid(row=row, column=col, padx=BUTTON_PADX, pady=BUTTON_PADY, sticky='ew')

        # Configure grid weights
        ops_frame.columnconfigure(0, weight=1)
        ops_frame.columnconfigure(1, weight=1)

    def create_generation_section(self):
        """Create improved signal generation section"""
        # Scalable parameters
        TITLE_FONT_SIZE = 9
        TITLE_PADY = 2
        FRAME_PADX = 5
        FRAME_PADY = 2
        FRAME_IPADX = 2
        FRAME_IPADY = 2
        LABEL_WIDTH = 6
        LABEL_FONT_SIZE = 8
        ENTRY_WIDTH = 8
        BUTTON_WIDTH = 12
        BUTTON_FONT_SIZE = 8
        ROW_PADY = 1
        PARAMS_PADY = 2
        BUTTON_PADY = 3

        # Title
        gen_label = tk.Label(self.main_frame, text="Generate Signal",
                             font=("Arial", TITLE_FONT_SIZE, "bold"), bg='light gray')
        gen_label.pack(pady=TITLE_PADY)

        # Generation frame with border
        gen_frame = tk.LabelFrame(self.main_frame, text="Parameters",
                                  bg='light gray', font=("Arial", LABEL_FONT_SIZE),
                                  relief=tk.GROOVE, borderwidth=1)
        gen_frame.pack(fill=tk.X, padx=FRAME_PADX, pady=FRAME_PADY, ipadx=FRAME_IPADX, ipady=FRAME_IPADY)

        # Wave type selection
        wave_frame = tk.Frame(gen_frame, bg='light gray')
        wave_frame.pack(fill=tk.X, pady=ROW_PADY)
        tk.Label(wave_frame, text="Type:", bg='light gray', width=LABEL_WIDTH,
                 anchor='w', font=("Arial", LABEL_FONT_SIZE)).pack(side=tk.LEFT)
        self.wave_type = tk.StringVar(value="Sine")
        wave_menu = tk.OptionMenu(wave_frame, self.wave_type, "Sine", "Cosine")
        wave_menu.config(width=ENTRY_WIDTH, font=("Arial", LABEL_FONT_SIZE))
        wave_menu.pack(side=tk.LEFT)

        # Input parameters in a neat grid
        params_frame = tk.Frame(gen_frame, bg='light gray')
        params_frame.pack(fill=tk.X, pady=PARAMS_PADY)

        # Row 1: Amplitude and Frequency
        row1 = tk.Frame(params_frame, bg='light gray')
        row1.pack(fill=tk.X, pady=ROW_PADY)

        # Amplitude
        tk.Label(row1, text="Amplitude:", bg='light gray', width=LABEL_WIDTH + 2,
                 anchor='w', font=("Arial", LABEL_FONT_SIZE)).pack(side=tk.LEFT)
        self.amplitude_var = tk.StringVar(value="1.0")
        amp_entry = tk.Entry(row1, textvariable=self.amplitude_var,
                             width=ENTRY_WIDTH, font=("Arial", LABEL_FONT_SIZE))
        amp_entry.pack(side=tk.LEFT, padx=2)

        # Frequency
        tk.Label(row1, text="Freq:", bg='light gray', width=LABEL_WIDTH - 2,
                 anchor='w', font=("Arial", LABEL_FONT_SIZE)).pack(side=tk.LEFT, padx=(10, 0))
        self.freq_var = tk.StringVar(value="1.0")
        freq_entry = tk.Entry(row1, textvariable=self.freq_var,
                              width=ENTRY_WIDTH, font=("Arial", LABEL_FONT_SIZE))
        freq_entry.pack(side=tk.LEFT, padx=2)

        # Row 2: Phase and Sampling Frequency
        row2 = tk.Frame(params_frame, bg='light gray')
        row2.pack(fill=tk.X, pady=ROW_PADY)

        # Phase
        tk.Label(row2, text="Phase (°):", bg='light gray', width=LABEL_WIDTH + 2,
                 anchor='w', font=("Arial", LABEL_FONT_SIZE)).pack(side=tk.LEFT)
        self.phase_var = tk.StringVar(value="0.0")
        phase_entry = tk.Entry(row2, textvariable=self.phase_var,
                               width=ENTRY_WIDTH, font=("Arial", LABEL_FONT_SIZE))
        phase_entry.pack(side=tk.LEFT, padx=2)

        # Sampling Frequency
        tk.Label(row2, text="Fs:", bg='light gray', width=LABEL_WIDTH - 2,
                 anchor='w', font=("Arial", LABEL_FONT_SIZE)).pack(side=tk.LEFT, padx=(10, 0))
        self.sampling_var = tk.StringVar(value="100.0")
        sampling_entry = tk.Entry(row2, textvariable=self.sampling_var,
                                  width=ENTRY_WIDTH, font=("Arial", LABEL_FONT_SIZE))
        sampling_entry.pack(side=tk.LEFT, padx=2)

        # Generate button
        gen_btn = tk.Button(gen_frame, text="Generate Signal",
                            command=self.generate_wave,
                            width=BUTTON_WIDTH, font=("Arial", BUTTON_FONT_SIZE, "bold"),
                            bg='#e0e0e0', relief=tk.RAISED)
        gen_btn.pack(pady=BUTTON_PADY)

    def generate_wave(self):
        """Handle wave generation from input fields"""
        try:
            # Get values from input fields
            amplitude = float(self.amplitude_var.get())
            analog_freq = float(self.freq_var.get())
            phase_deg = float(self.phase_var.get())
            sampling_freq = float(self.sampling_var.get())
            is_cosine = (self.wave_type.get() == "Cosine")

            # Validate inputs
            if amplitude <= 0:
                messagebox.showerror("Error", "Amplitude must be positive")
                return
            if analog_freq <= 0:
                messagebox.showerror("Error", "Frequency must be positive")
                return
            if sampling_freq <= 0:
                messagebox.showerror("Error", "Sampling frequency must be positive")
                return

            # Execute generation
            params = {
                'amplitude': amplitude,
                'analog_freq': analog_freq,
                'phase_deg': phase_deg,
                'sampling_freq': sampling_freq,
                'is_cosine': is_cosine
            }

            success = self.gui.execute_operation("sinusoidal", **params)
            if success:
                # Update signal list after generation
                self.update_signal_list()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input value:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Generation failed:\n{str(e)}")

    def update_signal_list(self):
        """Update the signal treeview"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Clear signal variables
        self.signal_vars.clear()

        # Add signals to treeview
        for i, signal in enumerate(self.gui.data_manager.get_all_signals()):
            signal_id = f"signal_{i}"

            # Create checkbutton variables
            cont_var = tk.BooleanVar()
            disc_var = tk.BooleanVar()
            self.signal_vars[signal_id] = {'continuous': cont_var, 'discrete': disc_var}

            # Bind variables to update display
            cont_var.trace('w', lambda *args, sid=signal_id: self.on_display_change(sid))
            disc_var.trace('w', lambda *args, sid=signal_id: self.on_display_change(sid))

            # Insert into treeview
            self.tree.insert('', 'end', iid=signal_id, values=(
                signal['name'][:25] + "..." if len(signal['name']) > 25 else signal['name'],
                '☐',  # Placeholder for continuous checkbox
                '☐'   # Placeholder for discrete checkbox
            ))

    def on_tree_click(self, event):
        """Handle clicks on treeview (for checkboxes)"""
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        if item and column in ('#2', '#3'):  # Continuous or Discrete column
            if column == '#2':  # Continuous column
                var = self.signal_vars[item]['continuous']
                var.set(not var.get())
            elif column == '#3':  # Discrete column
                var = self.signal_vars[item]['discrete']
                var.set(not var.get())

            self.update_checkbox_display(item)

    def update_checkbox_display(self, item):
        """Update checkbox display in treeview"""
        cont_var = self.signal_vars[item]['continuous']
        disc_var = self.signal_vars[item]['discrete']

        cont_char = '☑' if cont_var.get() else '☐'
        disc_char = '☑' if disc_var.get() else '☐'

        current_values = self.tree.item(item)['values']
        self.tree.item(item, values=(current_values[0], cont_char, disc_char))

    def on_display_change(self, signal_id):
        """Handle display mode changes"""
        self.update_checkbox_display(signal_id)
        self.gui.update_plot_display()

    def get_display_modes(self):
        """Get display modes for all signals"""
        display_modes = {}
        for signal_id, vars_dict in self.signal_vars.items():
            index = int(signal_id.split('_')[1])
            display_modes[index] = {
                'continuous': vars_dict['continuous'].get(),
                'discrete': vars_dict['discrete'].get()
            }
        return display_modes

    def get_selected_signals(self):
        """Get currently selected signals in treeview"""
        selection = self.tree.selection()
        if not selection:
            return []
        return [int(item_id.split('_')[1]) for item_id in selection]