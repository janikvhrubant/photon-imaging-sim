# here i want to generate a class which setups the sampling class according to the respective part of the configuration

import numpy as np
from src.utils.samplers import MCSampler, SobolSampler, HaltonSampler
from src.phantom.water_wall import WaterWall
from src.simulation.photon_generator import PhotonGenerator
from src.utils.logger import setup_logger

class SamplerSetup:
    def __init__(self, config):
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Initializing SamplerSetup with configuration.")
        assert 'SAMPLER' in config, self.logger.error("Configuration must contain 'SAMPLER' section.")
        self.logger.info(f"Configuration: {config['SAMPLER']}")
        self.config = config['SAMPLER']
        self.sampler = self._setup_sampler()

    def _setup_sampler(self):
        num_photons = self.config['num_photons']
        num_scatters = self.config.get('num_scatters', 0)
        sampler_type = self.config.get('sampling_strategy', 'MC')

        match sampler_type:
            case 'MC':
                self.logger.info("Setting up Monte Carlo sampler.")
                return MCSampler(num_photons, num_scatters)
            case 'Sobol':
                self.logger.info("Setting up Sobol sampler.")
                return SobolSampler(num_photons, num_scatters)
            case 'Halton':
                self.logger.info("Setting up Halton sampler.")
                return HaltonSampler(num_photons, num_scatters)
            case _:
                self.logger.error(f"Unknown sampling strategy: {sampler_type}. Defaulting to Monte Carlo sampler.")
                return MCSampler(num_photons, num_scatters)
            

class PhantomSetup:
    def __init__(self, config):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = config['PHANTOM']
        assert 'PHANTOM' in config, self.logger.error("Configuration must contain 'PHANTOM' section.")
        self.logger.info(f"Configuration: {self.config}")
        self.phantom = self._setup_phantom()

    def _setup_phantom(self):
        phantom_type = self.config.get('type', 'WaterWall')

        match phantom_type:
            case 'WaterWall':
                return WaterWall(
                    shape=tuple(self.config['shape']),
                    wall_thickness=self.config.get('wall_thickness', 5.0),
                    voxel_size=self.config.get('voxel_size', 1.0),
                    detector_distance=self.config.get('detector_distance', 0.0)
                )
            case _:
                self.logger.error(f"Unknown phantom type: {phantom_type}. Defaulting to WaterWall.")

class PhotonGeneratorSetup:
    def __init__(self, config):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = config['XRAY_TUBE']
        assert 'XRAY_TUBE' in config, self.logger.error("Configuration must contain 'XRAY_TUBE' section.")
        self.logger.info(f"Configuration: {self.config}")
        self.photon_generator = self._setup_photon_generator()
        
    def _setup_photon_generator(self):
        return PhotonGenerator(
            tube_voltage=float(self.config['tube_voltage']),
            anode_material=self.config.get('anode_material', 'W'),
            filter=[next(iter(item.items())) for item in self.config['filter']],
            angle=self.config.get('angle', None)
        )