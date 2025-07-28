import numpy as np
from src.utils.logger import setup_logger

class Detector:
    def __init__(self, grid):
        self.detector_grid = np.zeros((grid.shape[0], grid.shape[1]), dtype=float)
        self.detector_z_axis = grid.shape[2]
        self.logger = setup_logger(self.__class__.__name__)

    def add_signal(self, positions, signals):
        # generate a mask of positions where the z-coordinate is equal to the detector z-axis
        mask = np.isclose(positions[:, 2], self.detector_z_axis)
        self.logger.info(f"{np.sum(~mask)} photons did not hit the detector!")
        relevant_positions = positions[mask][:, :2]
        relevant_positions_idx = np.floor(relevant_positions).astype(int)
        relevant_signals = signals[mask]
        for pos, signal in zip(relevant_positions_idx, relevant_signals):
            if 0 <= pos[0] < self.detector_grid.shape[0] and 0 <= pos[1] < self.detector_grid.shape[1]:
                self.detector_grid[pos[0], pos[1]] += signal
            else:
                self.logger.warning(f"Position {pos} is out of bounds for the detector grid with shape {self.detector_grid.shape}. Signal {signal} not added.")
