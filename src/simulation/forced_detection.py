from src.utils.logger import setup_logger
import numpy as np

class ForcedDetection:
    def __init__(self, grid: np.ndarray, source_positions: np.ndarray,
                 directions: np.ndarray, energies: np.ndarray, randoms: np.ndarray):
        self.grid = grid
        self.source_positions = source_positions
        self.directions = directions
        self.energies = energies
        self.randoms = randoms
        self.logger = setup_logger(self.__class__.__name__)
        
        self.logger.info(f"ForcedDetection initialized with {len(source_positions)} sources, "
                         f"{len(directions)} directions, and {len(energies)} energies.")
        
    