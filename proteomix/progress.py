"""
Progress bars for long runs.

Progress uses tqdm when it is installed (pip install tqdm) and otherwise draws a simple
text bar on the terminal. Nothing is drawn when output is not a terminal (e.g. redirected
to a file or run on a cluster); the run log records every step either way.

    with Progress(total=5, desc="Analysis") as bar:
        bar.step("Normalising")
        bar.add(3)                     # more steps discovered later
"""
import logging
import shutil
import sys
import time

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:                                   # pragma: no cover
    _tqdm = None

_log = logging.getLogger("proteomix")
ACTIVE = []          # bars currently drawn; console log lines are printed through them


class Progress:
    def __init__(self, total: int, desc: str = "proteomix", enabled: bool = True):
        self.total, self.n, self.desc = max(int(total), 1), 0, desc
        self.enabled = enabled and sys.stderr.isatty()
        self._bar = None
        self._t0 = time.time()
        self._msg = ""
        if self.enabled and _tqdm is not None:
            self._bar = _tqdm(total=self.total, desc=desc, unit="step", leave=True,
                              dynamic_ncols=True, file=sys.stderr)
        if self.enabled:
            ACTIVE.append(self)

    # -- API ---------------------------------------------------------------------------
    def step(self, message: str = "", n: int = 1):
        """Advances the bar by n and shows message; the message also goes to the run log."""
        self.n += n
        self._msg = message
        if message:
            _log.debug(f"[{self.desc}] {message}")
        if self._bar is not None:
            self._bar.set_postfix_str(message[:50], refresh=False)
            self._bar.update(n)
        elif self.enabled:
            self._draw()

    def add(self, n: int):
        """Increases the total number of steps (when more work becomes known)."""
        self.total += int(n)
        if self._bar is not None:
            self._bar.total = self.total
            self._bar.refresh()

    def write(self, text: str):
        """Prints a line without breaking the bar."""
        if self._bar is not None:
            self._bar.write(text, file=sys.stdout)
        else:
            if self.enabled:
                sys.stderr.write("\r" + " " * shutil.get_terminal_size((80, 20)).columns + "\r")
            print(text, flush=True)
            if self.enabled:
                self._draw()

    def close(self):
        if self in ACTIVE:
            ACTIVE.remove(self)
        if self._bar is not None:
            self._bar.close()
        elif self.enabled:
            self._draw()
            sys.stderr.write("\n")
            sys.stderr.flush()
        self.enabled = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- fallback text bar ---------------------------------------------------------------
    def _draw(self):
        width = shutil.get_terminal_size((80, 20)).columns
        frac = min(self.n / self.total, 1.0)
        elapsed = time.time() - self._t0
        head = f"{self.desc}: {frac * 100:3.0f}% "
        tail = f" {self.n}/{self.total} [{elapsed:4.0f}s] {self._msg}"
        bar_w = max(10, min(30, width - len(head) - len(tail) - 3))
        filled = int(round(bar_w * frac))
        line = f"{head}|{'█' * filled}{' ' * (bar_w - filled)}|{tail}"
        sys.stderr.write("\r" + line[: width - 1].ljust(width - 1))
        sys.stderr.flush()


class NullProgress:
    """Stand-in used when no progress bar is wanted."""
    total = n = 0

    def step(self, message: str = "", n: int = 1):
        if message:
            _log.debug(message)

    def add(self, n: int):
        pass

    def write(self, text: str):
        print(text, flush=True)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
