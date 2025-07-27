import numpy as np
from abc import ABC, abstractmethod
from ..utils.logger import setup_logger
from scipy.stats import qmc

class Sampler(ABC):
    def __init__(self, num_photons: int, num_scatters: int):
        self.num_photons = int(float(num_photons))
        self.num_scatters = int(num_scatters)
        self.logger = setup_logger(self.__class__.__name__)
        self.randoms = self.sample()
        if self.num_scatters == 0:
            self.logger.info(f"Sampling {self.num_photons} photons with no scattering.")

    @abstractmethod
    def sample(self) -> np.ndarray:
        """Sample photons based on the specific sampling method."""
        pass

class MCSampler(Sampler):
    def sample(self) -> np.ndarray:
        """Sample photons using Monte Carlo method."""
        self.logger.info(f"Sampling {self.num_photons} photons using Monte Carlo method.")
        rng = np.random.default_rng()
        return rng.random((self.num_photons, 3 * (1 + self.num_scatters)))
    
class SobolSampler(Sampler):
    def sample(self) -> np.ndarray:
        """Sample photons using Sobol sequence."""
        self.logger.info(f"Sampling {self.num_photons} photons using Sobol sequence.")
        sampler = qmc.Sobol(d=3*(1+self.num_scatters), scramble=True)
        return sampler.random(n=self.num_photons)

class HaltonSampler(Sampler):
    def sample(self) -> np.ndarray:
        """Sample photons using Halton sequence."""
        self.logger.info(f"Sampling {self.num_photons} photons using Halton sequence.")
        sampler = qmc.Halton(d=3*(1+self.num_scatters), scramble=True)
        return sampler.random(n=self.num_photons)
