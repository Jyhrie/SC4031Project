import tkinter as tk
import threading


class SmartHomeUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Home Node")
        self.root.geometry("400x200")

        self.label = tk.Label(
            self.root,
            text="Waiting for commands...",
            font=("Arial", 14),
            fg="blue"
        )
        self.label.pack(expand=True)

        self._clear_job = None

    def start(self):
        self.root.mainloop()

    def show_command(self, action, device, a_conf=None, d_conf=None):
        text = f"{action} → {device}"

        if a_conf is not None:
            text += f"\nA:{a_conf:.2f} D:{d_conf:.2f}"

        self.root.after(0, self._update, text)

    def _update(self, text):
        self.label.config(text=text, fg="green")

        if self._clear_job:
            self.root.after_cancel(self._clear_job)

        self._clear_job = self.root.after(3000, self._clear)

    def _clear(self):
        self.label.config(text="Waiting for commands...", fg="blue")
        self._clear_job = None


ui = SmartHomeUI()