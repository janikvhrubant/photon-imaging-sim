import numpy as np
from ..utils.logger import setup_logger

from spekpy import Spek

class PhotonGenerator:
    def __init__(self, tube_voltage, anode_material = 'W', filter = [('Sn', 0.4), ('Cu', 0.1)],
                 angle: float | None = None, direction: np.ndarray | None = None):
        """
        Args:
            energy_distribution: Function to sample photon energy
            direction_distribution: Function to sample photon direction
        """
        self.spectrum = Spek(
            kvp=tube_voltage,
            targ=anode_material
        )
        self.spectrum = self.spectrum.multi_filter(filter)
        
        self.energies, probs = self.spectrum.get_spectrum()
        self.cdf = np.cumsum(probs / np.sum(probs))
        self.logger = setup_logger(self.__class__.__name__)
        self.beam_angle = angle
        if direction is None:
            self.beam_direction = np.array([0, 0, 1])
            self.logger.info(f"Using default beam direction: {self.beam_direction}")
        else:
            if not isinstance(direction, np.ndarray):
                self.logger.error("Direction must be a numpy array.")
            if direction.shape != (3,):
                self.logger.error("Direction must be a 3-element vector.")
            self.logger.info(f"Using given beam direction: {direction}")
        self.beam_direction = self.beam_direction / np.linalg.norm(self.beam_direction)


    def sample_energies(self, randoms):
        assert randoms.ndim == 1, self.logger.error("Random numbers must be a 1D array with shape (1, N).")
        self.logger.info(f"Sampling energy for {len(randoms)} photons.")
        idx = np.searchsorted(self.cdf, np.asarray(randoms), side='right')
        return self.energies[idx]
    

    def sample_directions(self, randoms):
        assert randoms.ndim == 2 and randoms.shape[1] == 2, \
            self.logger.error("Random numbers must be a 2D array with shape (2, N).")
        
        N = randoms.shape[1]
        self.logger.info(f"Sampling directions for {N} photons.")
        cos_alpha = np.cos(self.beam_angle)

        cos_theta = (1 - cos_alpha) * randoms[:,0] + cos_alpha
        sin_theta = np.sqrt(1 - cos_theta**2)

        phi = 2 * np.pi * randoms[:,1]

        x = sin_theta * np.cos(phi)
        y = sin_theta * np.sin(phi)
        z = cos_theta

        if np.allclose(self.beam_direction, [0, 0, 1]):
            return np.column_stack((x, y, z))
        
        axis = np.cross([0, 0, 1], self.beam_direction)
        axis_len = np.linalg.norm(axis)

        if axis_len < 1e-8:
            return np.column_stack((x, y, z)) if np.dot([0, 0, 1], self.beam_direction) > 0 else -np.column_stack((x, y, z))

        assert randoms.ndim == 2 and randoms.shape[1] == 2, self.logger.error("Random numbers must be a 2D array with shape (N, 2).")
        self.logger.info(f"Sampling directions for {randoms.shape[0]} photons.")
        cos_alpha = np.cos(self.alpha)

        cos_theta = (1 - cos_alpha) * randoms[:,0] + cos_alpha
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = 2 * np.pi * randoms[:,1]

        directions = np.stack([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ], axis=-1)

        if np.allclose(self.beam_direction, [0, 0, 1]):
            return directions
        
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, self.beam_direction)
        c = np.dot(z_axis, self.beam_direction)

        if np.allclose(v, 0):
            return directions if c > 0 else -directions
        
        s = np.linalg.norm(v)
        kmat = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + kmat + kmat @ kmat * ((1-c) / (s**2))
        
        return directions @ R.T

        
    def update_angle(self, world) -> float:
        grid = world.grid
        source_position = world.source_position
        grid_shape = grid.shape
        if self.beam_angle is None:
            length_distance = grid_shape[2]
            width_distance = np.max(grid_shape[0:2] - source_position[0:2])
            self.beam_angle = np.arctan2(width_distance, length_distance)
            self.logger.info(f"Beam angle set to {self.beam_angle:.2f} radians")
            self.beam_angle = self.beam_angle
        else:
            self.logger.info("Beam angle already set, skipping update.")
