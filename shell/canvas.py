"""
shell/canvas.py
ANSI terminal rendering utilities for the Neuryx shell.

Zero external dependencies — pure Python escape codes.
"""

from __future__ import annotations
import os
import sys
import time
import shutil

# ── ANSI escape codes ─────────────────────────────────────────────────────────

RST = "\033[0m"

class A:   # Attribute namespace
    BOLD  = "\033[1m"
    DIM   = "\033[2m"
    ITAL  = "\033[3m"
    UNDER = "\033[4m"
    BLINK = "\033[5m"

class F:   # Foreground colour namespace
    BLACK   = "\033[30m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"
    ORANGE  = "\033[38;5;208m"
    TEAL    = "\033[38;5;43m"
    VIOLET  = "\033[38;5;135m"
    PINK    = "\033[38;5;205m"


def paint(text: str, *codes: str) -> str:
    """Wrap text with ANSI codes, reset at end."""
    return "".join(codes) + str(text) + RST

# Convenience wrappers
def bold(t):    return paint(t, A.BOLD)
def dim(t):     return paint(t, A.DIM)
def green(t):   return paint(t, F.GREEN,   A.BOLD)
def red(t):     return paint(t, F.RED,     A.BOLD)
def yellow(t):  return paint(t, F.YELLOW)
def cyan(t):    return paint(t, F.CYAN,    A.BOLD)
def magenta(t): return paint(t, F.MAGENTA, A.BOLD)
def grey(t):    return paint(t, F.GREY)
def teal(t):    return paint(t, F.TEAL,    A.BOLD)
def violet(t):  return paint(t, F.VIOLET,  A.BOLD)
def orange(t):  return paint(t, F.ORANGE)
def white(t):   return paint(t, F.WHITE,   A.BOLD)

# ── Terminal dimensions ───────────────────────────────────────────────────────

def term_width() -> int:
    return min(shutil.get_terminal_size((80, 24)).columns, 88)

W = term_width()

# ── Layout primitives ─────────────────────────────────────────────────────────

def rule(char: str = "─", color: str = F.GREY, width: int | None = None) -> None:
    w = width or W
    print(paint(char * w, color))


def double_rule(color: str = F.CYAN) -> None:
    print(paint("═" * W, color))


def blank() -> None:
    print()


def heading(title: str, subtitle: str = "", color: str = F.CYAN) -> None:
    double_rule(color)
    print(paint(f"  {title}", color, A.BOLD))
    if subtitle:
        print(paint(f"  {subtitle}", F.GREY))
    double_rule(color)


def section_open(label: str, color: str = F.VIOLET) -> None:
    blank()
    tail = "─" * max(0, W - len(label) - 7)
    print(paint(f"┌─  {label}  {tail}┐", color, A.BOLD))


def section_close(color: str = F.VIOLET) -> None:
    print(paint("└" + "─" * (W - 1) + "┘", color))


def kv(label: str, value, label_w: int = 26) -> None:
    lbl = paint(f"  {label:<{label_w}}", F.GREY)
    val = paint(str(value), F.WHITE)
    print(f"{lbl} {val}")


def bullet(label: str, value) -> None:
    print(f"  {teal('◆')} {grey(label + ':')} {white(str(value))}")


def ok(msg: str) -> None:
    print(f"  {green('✓')}  {msg}")


def warn(msg: str) -> None:
    print(f"  {yellow('⚠')}  {yellow(msg)}")


def err(msg: str) -> None:
    print(f"  {red('✗')}  {red(msg)}")


def info_line(msg: str) -> None:
    print(f"  {grey('·')}  {grey(msg)}")


# ── Progress bar ──────────────────────────────────────────────────────────────

_BAR_WIDTH = 36

def progress(current: int, total: int, label: str = "") -> None:
    pct    = current / max(total, 1)
    filled = int(pct * _BAR_WIDTH)
    bar    = paint("█" * filled, F.TEAL) + paint("░" * (_BAR_WIDTH - filled), F.GREY)
    pct_s  = paint(f"{pct*100:5.1f}%", F.YELLOW)
    cnt_s  = paint(f"{current:>{len(str(total))}}/{total}", F.GREY)
    sys.stdout.write(f"  [{bar}] {pct_s}  {cnt_s}  {label}\r")
    sys.stdout.flush()


def progress_done() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


# ── Spinner ───────────────────────────────────────────────────────────────────

_SPIN_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_spin_idx = 0

def spin_tick(label: str = "") -> None:
    global _spin_idx
    frame = paint(_SPIN_FRAMES[_spin_idx % len(_SPIN_FRAMES)], F.CYAN)
    sys.stdout.write(f"\r  {frame}  {label}  ")
    sys.stdout.flush()
    _spin_idx += 1


# ── Format tables ─────────────────────────────────────────────────────────────

def format_menu(options: dict[int, str]) -> None:
    """Render a numbered menu with teal bullets."""
    for key, label in sorted(options.items()):
        num   = paint(f"  [{key}]", F.TEAL, A.BOLD)
        desc  = paint(label, F.WHITE)
        print(f"{num}  {desc}")


# ── File prompt ───────────────────────────────────────────────────────────────

def ask_file(label: str, allowed_exts: list[str] | None = None) -> str:
    """
    Prompt the user for a file path, validate it exists.
    Retries until a valid path is given.
    """
    ext_hint = ""
    if allowed_exts:
        ext_hint = "  " + grey("(" + " | ".join(allowed_exts) + ")")

    while True:
        try:
            path = input(f"\n  {teal('▶')} {label}{ext_hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            blank()
            err("Input cancelled.")
            raise SystemExit(0)

        if not path:
            warn("No path entered — try again.")
            continue

        # Try common extensions if missing
        if allowed_exts and not any(path.lower().endswith(e) for e in allowed_exts):
            for ext in allowed_exts:
                if os.path.exists(path + ext):
                    path = path + ext
                    break

        if os.path.exists(path):
            return path

        err(f"File not found: '{path}'")
        info_line("Tip: drag & drop the file into the terminal to paste its full path.")


def ask_int(prompt: str, choices: list[int]) -> int:
    """Prompt for an integer within the allowed choices. Retries on invalid input."""
    while True:
        try:
            raw = input(f"\n  {teal('▶')} {prompt}: ").strip()
            val = int(raw)
            if val in choices:
                return val
            warn(f"Please enter one of: {choices}")
        except (ValueError, TypeError):
            warn("Invalid input — enter a number.")
        except (EOFError, KeyboardInterrupt):
            blank()
            raise SystemExit(0)


def ask_str(prompt: str, default: str = "") -> str:
    """Prompt for a string, returning default if blank."""
    hint = f" [{default}]" if default else ""
    try:
        raw = input(f"\n  {teal('▶')} {prompt}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        blank()
        raise SystemExit(0)
    return raw if raw else default


# ── Neuryx ASCII logo ─────────────────────────────────────────────────────────

LOGO = rf"""
{paint('', F.TEAL, A.BOLD)}
  ███╗   ██╗███████╗██╗   ██╗██████╗ ██╗   ██╗██╗  ██╗
  ████╗  ██║██╔════╝██║   ██║██╔══██╗╚██╗ ██╔╝╚██╗██╔╝
  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝ ╚████╔╝  ╚███╔╝
  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗  ╚██╔╝   ██╔██╗
  ██║ ╚████║███████╗╚██████╔╝██║  ██║   ██║   ██╔╝ ██╗
  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
{RST}{paint('  General-Purpose Neural Sequence Engine', F.GREY)}   {paint('v2.0.0', F.TEAL)}
{paint('  Decoder-only Transformer  ·  Pure Python  ·  Zero ML dependencies', F.GREY)}
"""
