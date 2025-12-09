"""
Main Entry Point for Time Domain GUI
Run this file to start the Time Domain Operations application
"""
import tkinter as tk
from time_domain_gui import TimeDomainGUI


def main():
    """Main function"""
    root = tk.Tk()
    app = TimeDomainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()