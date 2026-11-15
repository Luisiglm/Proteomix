"""
Run logging: a detailed, timestamped run.log in the output folder (parameters, software
versions, input-file fingerprints, every analysis step and its duration, warnings) and short
messages on the console.

    logger = setup_run_logging("results")
    log_environment(logger); log_parameters(logger, vars(args)); log_inputs(logger, [path])
    with stage(logger, "Fitting model"):
        ...
"""
import hashlib
import logging
import os
import platform
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

LOGGER_NAME = "proteomix"


class _ConsoleHandler(logging.StreamHandler):
    """Console output that does not break an active tqdm bar."""
    def emit(self, record):
        from .progress import ACTIVE
        msg = self.format(record)
        if ACTIVE:
            ACTIVE[-1].write(msg)
        else:
            print(msg, flush=True)


def setup_run_logging(out_dir: str, filename: str = "run.log", console: bool = True,
                      console_level=logging.INFO) -> logging.Logger:
    """File handler (everything, with timestamps) + console handler (INFO and above)."""
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fh = logging.FileHandler(os.path.join(out_dir, filename), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    if console:
        ch = _ConsoleHandler()
        ch.setLevel(console_level)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    logger.propagate = False

    # route Python warnings (e.g. from the readers) into the log as well
    def _showwarning(message, category, filename_, lineno, file=None, line=None):
        logger.warning(f"{category.__name__}: {message}")
    warnings.showwarning = _showwarning
    return logger


def _version(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except ImportError:
        return "not installed"


def log_environment(logger):
    from . import __version__
    logger.debug("=" * 78)
    logger.debug("Environment")
    logger.debug(f"  proteomix   {__version__}")
    logger.debug(f"  python      {sys.version.split()[0]} ({platform.python_implementation()})")
    logger.debug(f"  platform    {platform.platform()}")
    for mod in ("torch", "numpy", "scipy", "pandas", "matplotlib", "tqdm"):
        logger.debug(f"  {mod:11s} {_version(mod)}")
    logger.debug(f"  command     {' '.join(sys.argv)}")
    logger.debug(f"  directory   {os.getcwd()}")
    logger.debug(f"  started     {datetime.now().isoformat(timespec='seconds')}")


def log_parameters(logger, params: dict, title: str = "Parameters"):
    logger.debug("=" * 78)
    logger.debug(title)
    width = max((len(k) for k in params), default=10)
    for k in sorted(params):
        logger.debug(f"  {k:<{width}}  {params[k]!r}")


def file_fingerprint(path: str) -> dict:
    """Size, modification time and SHA-256 of a file (to tie results to exact inputs)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = os.stat(path)
    return {"path": os.path.abspath(path), "bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "sha256": h.hexdigest()}


def log_inputs(logger, paths) -> list:
    logger.debug("=" * 78)
    logger.debug("Input files")
    out = []
    for p in paths:
        if p and os.path.exists(p):
            fp = file_fingerprint(p)
            out.append(fp)
            logger.debug(f"  {fp['path']}")
            logger.debug(f"      {fp['bytes']:,} bytes, modified {fp['modified']}, "
                         f"sha256 {fp['sha256']}")
    return out


@contextmanager
def stage(logger, name: str, progress=None):
    """Logs the start, end and duration of an analysis step; advances progress if given."""
    t0 = time.time()
    logger.debug(f"-- {name} ...")
    try:
        yield
    finally:
        logger.debug(f"-- {name} done ({time.time() - t0:.1f} s)")
        if progress is not None:
            progress.step(name)
