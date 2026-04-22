import tkinter as tk
import threading
from config import *

class SmartHomeUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Home Node (" + STATION_ID + ")")
        self.root.geometry("420x220")
        self.root.configure(bg="#1e1e1e")

        # STATE TITLE
        self.state_label = tk.Label(
            self.root,
            text="WAITING",
            font=("Arial", 20, "bold"),
            fg="#4ea1ff",
            bg="#1e1e1e"
        )
        self.state_label.pack(pady=15)

        # MAIN STATUS
        self.main_label = tk.Label(
            self.root,
            text="Listening for wake word...",
            font=("Arial", 14),
            fg="white",
            bg="#1e1e1e"
        )
        self.main_label.pack(pady=10)

        # DETAIL BOX
        self.detail_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 12),
            fg="#aaaaaa",
            bg="#1e1e1e"
        )
        self.detail_label.pack(pady=5)

        self._clear_job = None

    # ---------------- UI STATES ----------------

    def set_waiting(self):
        self.root.after(0, self._set_waiting_ui)

    def set_listening(self):
        self.root.after(0, self._set_listening_ui)

    def set_processing(self):
        self.root.after(0, self._set_processing_ui)

    def set_disconnected(self):
        self.root.after(0, self._set_disconnected_ui)

    def show_command(self, action, device, a_conf=None, d_conf=None):
        text = f"{action} → {device}"

        if a_conf is not None:
            text += f"\nA:{a_conf:.2f} D:{d_conf:.2f}"

        self.root.after(0, self._show_result, text)

    # ---------------- INTERNAL UI UPDATES ----------------

    def _set_waiting_ui(self):
        self.state_label.config(text="WAITING", fg="#4ea1ff")
        self.main_label.config(text="Listening for wake word...")
        self.detail_label.config(text="")

    def _set_listening_ui(self):
        self.state_label.config(text="LISTENING", fg="#ffd166")
        self.main_label.config(text="Capturing audio...")
        self.detail_label.config(text="")

    def _set_processing_ui(self):
        self.state_label.config(text="PROCESSING", fg="#06d6a0")
        self.main_label.config(text="Understanding command...")
        self.detail_label.config(text="Sending to server...")

    def _set_disconnected_ui(self):
        self.state_label.config(text="DISCONNECTED", fg="#06d6a0")
        self.main_label.config(text="reconnecting...")

    def _show_result(self, text):
        self.state_label.config(text="DONE", fg="#8aff80")
        self.main_label.config(text=text)
        self.detail_label.config(text="")

        if self._clear_job:
            self.root.after_cancel(self._clear_job)

        self._clear_job = self.root.after(3000, self._set_waiting_ui)

    # ---------------- MAIN LOOP ----------------

    def start(self):
        self.root.mainloop()

ui = SmartHomeUI()