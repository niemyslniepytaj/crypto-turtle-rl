"""
Karty ANSI w stylu Optuny — komunikaty PL, tagi EN.
"""

from __future__ import annotations

import os


def enable_windows_ansi():
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def colors(enabled=True):
    if not enabled or os.environ.get("NO_COLOR"):

        class _No:
            """Puste kody ANSI (wyłączone kolorowanie)."""
            reset = bold = dim = red = green = yellow = blue = magenta = cyan = ""

        return _No()

    class _Ansi:
        """Kody ANSI do kolorowego wyjścia konsoli."""
        reset = "\033[0m"
        bold = "\033[1m"
        dim = "\033[2m"
        red = "\033[31m"
        green = "\033[32m"
        yellow = "\033[33m"
        blue = "\033[34m"
        magenta = "\033[35m"
        cyan = "\033[36m"

    return _Ansi()


def print_card(title, lines, *, tag="TrainPPO", ok=None):
    """
    lines: lista (label, value) albo zwykłych stringów.
    ok: True/False/None → kolor tytułu.
    """
    enable_windows_ansi()
    c = colors()
    if ok is True:
        flag = f"{c.green}{c.bold}OK{c.reset}"
    elif ok is False:
        flag = f"{c.red}{c.bold}FAIL{c.reset}"
    else:
        flag = ""
    head = f"{c.cyan}{c.bold}[{tag}]{c.reset} {c.bold}{title}{c.reset}"
    if flag:
        head = f"{head}  {flag}"
    print()
    print(head)
    rule = "-" * 56  # ASCII — bezpieczne na cp1250 Windows
    print(f"{c.dim}{rule}{c.reset}")
    for item in lines:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            label, value = item[0], item[1]
            print(f"  {c.dim}{label:<22}{c.reset} {value}")
        else:
            print(f"  {item}")
    print(f"{c.dim}{rule}{c.reset}")


def print_run_summary(
    *,
    arm,
    seed,
    portfolio_mode,
    folds,
    run_dir,
    wall_times=None,
    status_counts=None,
):
    """I4.2 — czytelne podsumowanie runu po sekwencji."""
    lines = [
        ("ramię", arm),
        ("seed", seed),
        ("tryb", portfolio_mode),
        ("okna", ", ".join(str(f) for f in (folds or []))),
        ("run_dir", run_dir),
    ]
    if status_counts:
        lines.append(("statusy", status_counts))
    if wall_times is not None:
        lines.append(("wall_time_s (suma)", f"{float(wall_times):.1f}"))
    print_card("Podsumowanie runu", lines, tag="WF", ok=True)


def print_kv(tag, msg, *, kind="info"):
    enable_windows_ansi()
    c = colors()
    col = {
        "info": c.cyan,
        "ok": c.green,
        "warn": c.yellow,
        "err": c.red,
        "best": c.yellow,
    }.get(kind, c.cyan)
    print(f"{col}{c.bold}[{tag}]{c.reset} {msg}")
