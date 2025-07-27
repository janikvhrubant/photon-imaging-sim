import numpy as np

class Detector:
    def __init__(self, x_size: int, y_size: int):
        self.primary_intensity = np.zeros((x_size, y_size), dtype=np.float32)
        self.scatter_intensity = np.zeros((x_size, y_size), dtype=np.float32)

    def update_primary(self, intensities: np.ndarray):
        assert intensities.shape == self.primary_intensity.shape, \
            "Intensity shape mismatch"
        self.primary_intensity += intensities
        
    def update_scatter(self, intensities: np.ndarray):
        assert intensities.shape == self.scatter_intensity.shape, \
            "Intensity shape mismatch"
        self.scatter_intensity += intensities