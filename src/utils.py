"""Utility functions: unit conversions, logging, reproducibility, and timing."""
import os
import random
import time
import logging
from contextlib import contextmanager
from typing import Generator
import numpy as np


def setup_logger(name: str = "renewable_ei", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def mw_to_mwh(power_mw: float | np.ndarray, duration_hours: float = 1.0) -> float | np.ndarray:
    """Converts average power in MW over a duration in hours into energy in MWh.
    
    E (MWh) = P (MW) * dt (h)
    """
    return power_mw * duration_hours


def mwh_to_mw(energy_mwh: float | np.ndarray, duration_hours: float = 1.0) -> float | np.ndarray:
    """Converts energy in MWh over a duration in hours into average power in MW.
    
    P (MW) = E (MWh) / dt (h)
    """
    if duration_hours <= 0:
        raise ValueError("duration_hours must be strictly positive.")
    return energy_mwh / duration_hours


def calculate_cost(
    energy_mwh: float | np.ndarray,
    price_eur_mwh: float | np.ndarray,
    adder_eur_mwh: float = 0.0,
) -> float | np.ndarray:
    """Calculates procurement or settlement cost in EUR.
    
    Cost (EUR) = Energy (MWh) * (Price (EUR/MWh) + Adder (EUR/MWh))
    """
    return energy_mwh * (price_eur_mwh + adder_eur_mwh)


@contextmanager
def timer(description: str = "Execution") -> Generator[dict, None, None]:
    """Context manager for measuring elapsed execution time in seconds."""
    state = {"elapsed_seconds": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed_seconds"] = time.perf_counter() - start
