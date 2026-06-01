# -*- coding: utf-8 -*-
"""
definition_tester.py
====================
Interactive GUI for testing find_definition() from definition_finder.py.

Usage:
    python definition_tester.py

Controls:
    - Type or paste Swedish text in the top panel.
    - Click anywhere in the text to place the cursor on a word.
    - Press the "Look up word at cursor" button  OR  Ctrl+Enter.
    - The result and full debug log appear in the lower panels.
"""

import contextlib
import io
import re
import sys
import threading
import tkinter as tk
from tkinter import ttk

# Ensure the script can import definition_finder even when run from a
# different working directory, as long as both files share the same folder.
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from definition_finder import find_definition  # noqa: E402

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

BG          = "#1e1e2e"   # main background  (dark purple-grey)
BG_PANEL    = "#181825"   # widget interiors
ACCENT      = "#cba6f7"   # mauve
ACCENT2     = "#89b4fa"   # lavender-blue
SUCCESS     = "#a6e3a1"   # green
WARNING     = "#fab387"   # peach
ERROR       = "#f38ba8"   # red
FG          = "#cdd6f4"   # default text
FG_DIM      = "#585b70"   # subtle / placeholder
FG_LABEL    = "#bac2de"
BORDER      = "#313244"

FONT_FAMILY = "Segoe UI"
MONO_FAMILY = "Cascadia Code" if sys.platform == "win32" else "Courier"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_char_index(text_widget: tk.Text) -> int:
    """Convert the Tk INSERT cursor position to an absolute character offset."""
    pos = text_widget.index(tk.INSERT)          # e.g. "2.5"
    row, col = map(int, pos.split("."))
    content = text_widget.get("1.0", tk.END)    # always ends with "\n"
    lines = content.split("\n")
    # rows are 1-indexed in Tk
    char_idx = sum(len(lines[i]) + 1 for i in range(row - 1)) + col
    return char_idx


def _word_at(text: str, idx: int) -> str | None:
    """Return the word that contains character position *idx*, or None."""
    if not text or idx < 0 or idx > len(text):
        return None
    # Walk backwards to find the start of the word
    start = idx
    while start > 0 and re.match(r"\w", text[start - 1]):
        start -= 1
    # Walk forwards to find the end
    end = idx
    while end < len(text) and re.match(r"\w", text[end]):
        end += 1
    word = text[start:end]
    return word if word else None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class DefinitionTesterApp:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._lookup_thread: threading.Thread | None = None

        self._configure_root()
        self._build_styles()
        self._build_ui()
        self._bind_events()

    # ------------------------------------------------------------------ setup

    def _configure_root(self) -> None:
        self.root.title("Definition Finder Tester")
        self.root.configure(bg=BG)
        self.root.minsize(700, 680)
        self.root.geometry("860x780")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TFrame",       background=BG)
        style.configure("Panel.TFrame", background=BG_PANEL, relief="flat")

        style.configure(
            "Accent.TButton",
            background=ACCENT, foreground=BG,
            font=(FONT_FAMILY, 10, "bold"),
            padding=(14, 7),
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT2), ("disabled", BORDER)],
            foreground=[("disabled", FG_DIM)],
        )

        style.configure("TLabel",        background=BG,       foreground=FG_LABEL,
                        font=(FONT_FAMILY, 9))
        style.configure("Header.TLabel", background=BG,       foreground=FG,
                        font=(FONT_FAMILY, 10, "bold"))
        style.configure("Status.TLabel", background=BG_PANEL, foreground=FG_DIM,
                        font=(MONO_FAMILY, 9))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)

        row = 0

        # -- Input label ------------------------------------------------------
        ttk.Label(outer, text="Input text", style="Header.TLabel"
                  ).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        # -- Input text area --------------------------------------------------
        input_frame = ttk.Frame(outer, style="Panel.TFrame")
        input_frame.grid(row=row, column=0, sticky="nsew", pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)
        outer.rowconfigure(row, weight=3)
        row += 1

        self.input_text = tk.Text(
            input_frame,
            wrap="word",
            font=(FONT_FAMILY, 12),
            bg=BG_PANEL, fg=FG,
            insertbackground=ACCENT,
            selectbackground=ACCENT, selectforeground=BG,
            relief="flat", padx=12, pady=10,
            undo=True,
            highlightthickness=1,
            highlightcolor=ACCENT,
            highlightbackground=BORDER,
        )
        self.input_text.grid(row=0, column=0, sticky="nsew")

        input_scroll = ttk.Scrollbar(input_frame, orient="vertical",
                                     command=self.input_text.yview)
        input_scroll.grid(row=0, column=1, sticky="ns")
        self.input_text.configure(yscrollcommand=input_scroll.set)

        self._set_placeholder()

        # -- Toolbar row ------------------------------------------------------
        toolbar = ttk.Frame(outer)
        toolbar.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        toolbar.columnconfigure(1, weight=1)
        row += 1

        self.lookup_btn = ttk.Button(
            toolbar,
            text="Look up word at cursor",
            style="Accent.TButton",
            command=self._on_lookup,
        )
        self.lookup_btn.grid(row=0, column=0, sticky="w")

        self.status_var = tk.StringVar(
            value="Place cursor on a word, then look it up  |  Ctrl+Enter"
        )
        ttk.Label(toolbar, textvariable=self.status_var, style="Status.TLabel",
                  anchor="e").grid(row=0, column=1, sticky="e", padx=(12, 0))

        # -- Result label -----------------------------------------------------
        ttk.Label(outer, text="Result", style="Header.TLabel"
                  ).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        # -- Result panel -----------------------------------------------------
        result_frame = ttk.Frame(outer, style="Panel.TFrame")
        result_frame.grid(row=row, column=0, sticky="nsew", pady=(0, 12))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        outer.rowconfigure(row, weight=2)
        row += 1

        self.result_text = tk.Text(
            result_frame,
            wrap="word",
            font=(FONT_FAMILY, 11),
            bg=BG_PANEL, fg=FG,
            relief="flat", padx=12, pady=10,
            state="disabled",
            cursor="arrow",
            highlightthickness=1,
            highlightcolor=BORDER,
            highlightbackground=BORDER,
        )
        self.result_text.grid(row=0, column=0, sticky="nsew")

        result_scroll = ttk.Scrollbar(result_frame, orient="vertical",
                                      command=self.result_text.yview)
        result_scroll.grid(row=0, column=1, sticky="ns")
        self.result_text.configure(yscrollcommand=result_scroll.set)

        # Text tags for the result panel
        self.result_text.tag_configure("definition",
                                       font=(FONT_FAMILY, 13),
                                       foreground=FG)
        self.result_text.tag_configure("id_label",
                                       font=(FONT_FAMILY, 9),
                                       foreground=FG_DIM)
        self.result_text.tag_configure("id_value",
                                       font=(MONO_FAMILY, 9),
                                       foreground=ACCENT2)
        self.result_text.tag_configure("score_label",
                                       font=(FONT_FAMILY, 9),
                                       foreground=FG_DIM)
        self.result_text.tag_configure("score_value",
                                       font=(FONT_FAMILY, 9, "bold"),
                                       foreground=SUCCESS)
        self.result_text.tag_configure("none_msg",
                                       font=(FONT_FAMILY, 11, "italic"),
                                       foreground=ERROR)
        self.result_text.tag_configure("error_msg",
                                       font=(FONT_FAMILY, 10),
                                       foreground=ERROR)
        self.result_text.tag_configure("working",
                                       font=(FONT_FAMILY, 11, "italic"),
                                       foreground=WARNING)

        # -- Log label --------------------------------------------------------
        ttk.Label(outer, text="Log / debug output", style="Header.TLabel"
                  ).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1

        # -- Log panel --------------------------------------------------------
        log_frame = ttk.Frame(outer, style="Panel.TFrame")
        log_frame.grid(row=row, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        outer.rowconfigure(row, weight=2)

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            font=(MONO_FAMILY, 9),
            bg=BG_PANEL, fg=FG_DIM,
            relief="flat", padx=12, pady=10,
            state="disabled",
            cursor="arrow",
            highlightthickness=1,
            highlightcolor=BORDER,
            highlightbackground=BORDER,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical",
                                   command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.log_text.tag_configure("separator", foreground=BORDER)
        self.log_text.tag_configure("log_body",  foreground=FG_DIM)

    def _bind_events(self) -> None:
        self.input_text.bind("<KeyRelease>",      self._on_cursor_move)
        self.input_text.bind("<ButtonRelease-1>", self._on_cursor_move)
        self.root.bind("<Control-Return>", lambda _e: self._on_lookup())
        self.input_text.bind("<FocusIn>",  self._on_focus_in)
        self.input_text.bind("<FocusOut>", self._on_focus_out)

    # ---------------------------------------------------------- placeholder

    _PLACEHOLDER = "Paste or type Swedish text here..."
    _placeholder_active = True

    def _set_placeholder(self) -> None:
        self.input_text.insert("1.0", self._PLACEHOLDER)
        self.input_text.configure(fg=FG_DIM)
        self._placeholder_active = True

    def _on_focus_in(self, _event=None) -> None:
        if self._placeholder_active:
            self.input_text.delete("1.0", tk.END)
            self.input_text.configure(fg=FG)
            self._placeholder_active = False

    def _on_focus_out(self, _event=None) -> None:
        if not self.input_text.get("1.0", tk.END).strip():
            self._set_placeholder()

    # ---------------------------------------------------------- status bar

    def _on_cursor_move(self, _event=None) -> None:
        if self._placeholder_active:
            self.status_var.set(
                "Place cursor on a word, then look it up  |  Ctrl+Enter"
            )
            return
        content = self.input_text.get("1.0", tk.END)
        text = content.rstrip("\n")
        idx = _get_char_index(self.input_text)
        word = _word_at(text, idx) or "-"
        pos_str = self.input_text.index(tk.INSERT)
        self.status_var.set(
            "char {}  |  word: '{}'  |  pos {}".format(idx, word, pos_str)
        )

    # ---------------------------------------------------------- lookup

    def _on_lookup(self) -> None:
        if self._lookup_thread and self._lookup_thread.is_alive():
            return

        if self._placeholder_active:
            self.status_var.set("Please enter some text first.")
            return

        content = self.input_text.get("1.0", tk.END)
        sentence = content.rstrip("\n")
        if not sentence.strip():
            self.status_var.set("Text is empty.")
            return

        char_index = _get_char_index(self.input_text)
        word = _word_at(sentence, char_index)
        if not word:
            self.status_var.set("No word at cursor -- click on a word first.")
            return

        self.lookup_btn.configure(state="disabled")
        self.status_var.set("Looking up '{}' ...".format(word))
        self._write_result("working", "Looking up '{}' ...\n".format(word))
        self._append_log_separator(word, char_index)

        self._lookup_thread = threading.Thread(
            target=self._run_lookup,
            args=(sentence, word, char_index),
            daemon=True,
        )
        self._lookup_thread.start()

    def _run_lookup(self, sentence: str, word: str, char_index: int) -> None:
        """Runs in a background thread -- must not touch Tk widgets directly."""
        buf = io.StringIO()
        result = None
        error = None
        try:
            with contextlib.redirect_stdout(buf):
                result = find_definition(sentence, word, char_index)
        except Exception as exc:  # noqa: BLE001
            error = exc
        log_output = buf.getvalue()
        self.root.after(0, self._show_result, result, error, log_output, word)

    def _show_result(
        self,
        result: "dict | None",
        error: "Exception | None",
        log_output: str,
        word: str,
    ) -> None:
        """Called on the main thread once the background lookup finishes."""
        self.lookup_btn.configure(state="normal")

        if error is not None:
            self.status_var.set("Error: {}".format(error))
            self._write_result("error_msg", "Error:\n{}\n".format(error))
            self._append_log(log_output)
            return

        if result is None:
            self.status_var.set("No definition found for '{}'.".format(word))
            self._write_result("none_msg", "No definition found.\n")
            self._append_log(log_output)
            return

        defn  = result.get("definition", "")
        rid   = result.get("id", "")
        score = result.get("score", 0.0)
        self.status_var.set(
            "OK  '{}'  ->  {}  (score {:.4f})".format(word, rid, score)
        )
        self._write_result_rich(defn, rid, score)
        self._append_log(log_output)

    # ---------------------------------------------------------- panel writers

    def _write_result(self, tag: str, text: str) -> None:
        """Clear the result panel and write a single tagged message."""
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text, tag)
        self.result_text.configure(state="disabled")

    def _write_result_rich(self, definition: str, rid: str, score: float) -> None:
        """Render a successful result with styled fields."""
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)

        self.result_text.insert(tk.END, definition + "\n\n", "definition")
        self.result_text.insert(tk.END, "ID     ", "id_label")
        self.result_text.insert(tk.END, rid + "\n", "id_value")
        self.result_text.insert(tk.END, "Score  ", "score_label")
        self.result_text.insert(tk.END, "{:.4f}\n".format(score), "score_value")

        self.result_text.configure(state="disabled")

    def _append_log_separator(self, word: str, char_index: int) -> None:
        """Write a divider before each new lookup in the log."""
        self.log_text.configure(state="normal")
        sep = "{}\n  '{}' @ char {}\n{}\n".format(
            "-" * 60, word, char_index, "-" * 60
        )
        self.log_text.insert(tk.END, sep, "separator")
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        """Append captured stdout to the log panel and scroll to bottom."""
        if not text:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text, "log_body")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()

    # On Windows 11, try to colour the title bar to match the dark theme.
    try:
        import ctypes
        HWND = ctypes.windll.user32.GetForegroundWindow()
        color = int(BG.lstrip("#"), 16)
        # DWM expects BGR
        bgr = ((color & 0xFF) << 16) | (color & 0xFF00) | ((color >> 16) & 0xFF)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            HWND, 35, ctypes.byref(ctypes.c_int(bgr)), ctypes.sizeof(ctypes.c_int)
        )
    except Exception:
        pass

    _app = DefinitionTesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
