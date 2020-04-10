import tkinter as tk
from tkinter import filedialog


def LoadMovieFile():
    root = tk.Tk()
    root.withdraw()
    filePath = filedialog.askopenfilename(
        initialdir="C:",
        filetypes=([('Video Files', '*.mkv;*.avi;*.mp4;*.mpg;*.flv;*.wmv')]))
    return filePath

