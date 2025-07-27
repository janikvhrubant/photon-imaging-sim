import logging
import sys
from pathlib import Path

def setup_logger(name: str = "photon_sim", level=logging.INFO) -> logging.Logger:
    """
    Creates or retrieves a logger with the specified name.
    Ensures consistent formatting and console output.

    Args:
        name: Name of the logger (use __class__.__name__ in classes)
        level: Logging level (e.g., logging.DEBUG, logging.INFO)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def phantom_logger(log_path: str) -> logging.Logger:
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("voxel_model_description")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # verhindert Weiterleitung an Root-Logger

    if not logger.handlers:  # vermeidet doppelte Handler bei mehrfacher Initialisierung
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

