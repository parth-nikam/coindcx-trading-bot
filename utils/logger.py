import logging
import sys
from pathlib import Path

_loggers: dict = {}
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "nexus.log"
_file_handler: logging.FileHandler | None = None


def _get_file_handler() -> logging.FileHandler:
    global _file_handler
    if _file_handler is None:
        _LOG_DIR.mkdir(exist_ok=True)
        _file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        _file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        _file_handler.setLevel(logging.DEBUG)
    return _file_handler


def get_logger(name: str) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S"
        ))
        console.setLevel(logging.INFO)
        logger.addHandler(console)

        # File handler (shared across all loggers)
        logger.addHandler(_get_file_handler())
        logger.setLevel(logging.DEBUG)

    _loggers[name] = logger
    return logger
