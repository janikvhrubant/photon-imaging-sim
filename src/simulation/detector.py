import numpy as np
from src.utils.logger import setup_logger

class Detector:
    def __init__(self, grid, resolution_factor: int = 1):
        self.resolution_factor = resolution_factor
        grid_shape = (grid.shape[0]*resolution_factor, grid.shape[1]*resolution_factor)
        self.detector_grid = np.zeros((grid_shape[0], grid_shape[1]), dtype=float)
        self.detector_z_axis = grid.shape[2]
        self.logger = setup_logger(self.__class__.__name__)

    def add_signal(self, positions, signals):
        mask = np.isclose(positions[:, 2], self.detector_z_axis)
        self.logger.info(f"{np.sum(~mask)} photons did not hit the detector!")
        relevant_positions = positions[mask][:, :2]
        relevant_positions_idx = np.floor(relevant_positions*self.resolution_factor).astype(int)
        relevant_signals = signals[mask]
        for pos, signal in zip(relevant_positions_idx, relevant_signals):
            if 0 <= pos[0] < self.detector_grid.shape[0] and 0 <= pos[1] < self.detector_grid.shape[1]:
                self.detector_grid[pos[0], pos[1]] += signal
            else:
                self.logger.warning(f"Position {pos} is out of bounds for the detector grid with shape {self.detector_grid.shape}. Signal {signal} not added.")
