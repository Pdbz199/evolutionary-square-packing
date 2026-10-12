"""Evolutionary packing of N unit squares (rotation allowed) into the smallest square."""

from .ga import GAConfig, RunResult, run
from .records import best_known

__all__ = ["GAConfig", "RunResult", "run", "best_known"]
__version__ = "0.1.0"
