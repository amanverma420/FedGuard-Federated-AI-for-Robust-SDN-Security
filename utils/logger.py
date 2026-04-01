"""
utils/logger.py - Centralized logging for FedGuard
"""
import logging
import sys
from datetime import datetime

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-20s | %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger
