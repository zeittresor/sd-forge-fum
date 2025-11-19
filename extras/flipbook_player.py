#!/usr/bin/env python3

"""
flipbook_gui.py
===============

Devised by github.com/zeittresor
Code created using ollama:gpt-oss-safeguard:20b
Language: German

Ein minimalistisches Daumen‑Kino‑Programm.

* Select folder – wählt ein Ordner mit Bilddateien.
* View flipbook – startet das Vollbild‑Flip‑Book.
* Interval (ms) – Bildwechsel‑Zeit in Millisekunden (Standard 40 ms).
* ESC beendet das Flip‑Book (keine Konsole, wenn du pythonw.exe oder pyinstaller --noconsole verwendest).

Debug‑Modus:  python flipbook_gui.py -debug
"""

import sys
import logging
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

DEBUG = '-debug' in sys.argv
LOG_FILENAME = 'app.log'

if DEBUG:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(LOG_FILENAME, mode='w')]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(LOG_FILENAME, mode='w')]
    )

log = logging.getLogger(__name__)

def get_image_files(folder: Path) -> list[Path]:
    """Alle Bilddateien im Ordner (alphabetisch) zurückgeben."""
    exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    files.sort()
    log.debug(f'{len(files)} Bilddateien gefunden: {[p.name for p in files[:5]]} …')
    return files

def resize_to_screen(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Bild proportional auf Bildschirmgröße skalieren."""
    w, h = img.size
    ratio = min(max_w / w, max_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

class FlipBookApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flipbook Viewer")
        self.resizable(False, False)      # GUI soll nicht verkleinert werden

        self.folder_path: Path | None = None
        self.image_files: list[Path] = []
        self.interval_ms: int = 40
        self.flip_window: tk.Toplevel | None = None
        self.slider_dir_forward: bool = True
        self.current_index = 0
        self.photo_ref: ImageTk.PhotoImage | None = None      # Referenz auf Bild
        self.after_id = None

        self._build_gui()
        log.debug("GUI initialisiert")

    def _build_gui(self):
        """Erstellung der Widgets – mit `grid` für alles.  Keine Mischung aus
        `pack` und `grid` im selben Container."""
        pad = {'padx': 10, 'pady': 10}

        frm = ttk.Frame(self)
        frm.grid(row=0, column=0, sticky='nsew', **pad)

        ttk.Button(frm, text="Select folder",
                    command=self.select_folder).grid(row=0, column=0, **pad)

        ttk.Label(frm, text="Interval (ms):").grid(row=0, column=1, sticky='e')
        self.interval_var = tk.StringVar(value=str(self.interval_ms))
        ttk.Entry(frm, width=8, textvariable=self.interval_var).grid(row=0, column=2, **pad)

        self.btn_play = ttk.Button(frm, text="View flipbook",
                                    command=self.start_flipbook, state='disabled')
        self.btn_play.grid(row=0, column=3, **pad)

        self.status_lbl = ttk.Label(self, text="No folder selected",
                                    relief='sunken', anchor='w')
        self.status_lbl.grid(row=1, column=0, sticky='ew', padx=pad['padx'])

        self.columnconfigure(0, weight=1)

    def select_folder(self):
        path = filedialog.askdirectory()
        if not path:
            return

        self.folder_path = Path(path)
        files = get_image_files(self.folder_path)

        if not files:
            messagebox.showwarning("Kein Bild", "Im Ordner wurden keine Bilddateien gefunden.")
            self.image_files = []
            self.btn_play.config(state='disabled')
            self.status_lbl.config(text="No valid images")
            return

        self.image_files = files
        self.btn_play.config(state='normal')
        self.status_lbl.config(text=f"Folder: {self.folder_path.name} ({len(self.image_files)} Bilder)")

    def start_flipbook(self):
        if not self.image_files:
            messagebox.showerror("Fehler", "Bitte wähle zuerst einen Ordner mit Bildern aus.")
            return

        try:
            interval = int(float(self.interval_var.get()))
            if interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ungültig", "Bitte ein positives Zahlen‑Intervall eingeben.")
            return
        self.interval_ms = interval
        log.debug(f"Start Flipbook mit Intervall {self.interval_ms} ms")

        self.flip_window = tk.Toplevel(self)
        self.flip_window.attributes('-fullscreen', True)
        self.flip_window.protocol('WM_DELETE_WINDOW', self.exit_flipbook)  # Falls jemand X drückt
        self.flip_window.bind('<Escape>', self.exit_flipbook)

        self.lbl_image = ttk.Label(self.flip_window)
        self.lbl_image.pack(expand=True)

        self.current_index = 0
        self.slider_dir_forward = True
        self.show_image()
        self.after_id = self.flip_window.after(self.interval_ms, self.next_frame)

    def show_image(self):
        """Lädt das aktuelle Bild, skaliert es und zeigt es."""
        img_path = self.image_files[self.current_index]
        log.debug(f"Zeige Bild {self.current_index + 1}/{len(self.image_files)}: {img_path.name}")

        pil_img = Image.open(img_path)
        screen_w = self.flip_window.winfo_screenwidth()
        screen_h = self.flip_window.winfo_screenheight()
        pil_img = resize_to_screen(pil_img, screen_w, screen_h)

        self.photo_ref = ImageTk.PhotoImage(pil_img)
        self.lbl_image.configure(image=self.photo_ref)

    def next_frame(self):
        """Wechselt zum nächsten Bild (vorwärts + rückwärts)."""
        if self.slider_dir_forward:
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
            else:
                self.slider_dir_forward = False
                self.current_index -= 1
        else:
            if self.current_index > 0:
                self.current_index -= 1
            else:
                self.slider_dir_forward = True
                self.current_index += 1

        self.show_image()
        self.after_id = self.flip_window.after(self.interval_ms, self.next_frame)

    def exit_flipbook(self, *args):
        """ESC bzw. X beendet die Anzeige."""
        if self.flip_window:
            if self.after_id:
                try:
                    self.flip_window.after_cancel(self.after_id)
                except Exception:
                    pass
            self.flip_window.destroy()
            self.flip_window = None
        self.focus_set()   # Fokus zurück an Hauptfenster

def main():
    app = FlipBookApp()
    app.mainloop()

if __name__ == "__main__":
    main()
