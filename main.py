# main.py
from src.phantom.water_wall import WaterWall
from src.simulation.photon_generator import PhotonGenerator
import logging
import yaml
from src.simulation.grids import WorldGrid
import numpy as np
from src.utils.setup import SamplerSetup, PhantomSetup, PhotonGeneratorSetup
from src.simulation.detector import Detector
from src.simulation.attenuation import Attenuation
from src.utils.evaluation import Evaluation

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open('run_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # phantom_config = config['PHANTOM']
    # shape = np.array(phantom_config['shape']).astype(int)
    # wall_thickness = float(phantom_config['wall_thickness'])
    # voxel_size = float(phantom_config['voxel_size'])
    # detector_distance = float(phantom_config['detector_distance'])

    phantom = PhantomSetup(config).phantom
    
    world = WorldGrid(phantom.grid_path)
    # materials = world.get_all_materials()
    att = Attenuation(world.materials)

    scatter_detector = Detector(phantom.grid)
    primary_detector = Detector(phantom.grid)

    sampler = SamplerSetup(config).sampler

    photon_generator = PhotonGeneratorSetup(config).photon_generator
    photon_generator.update_angle(world)
    
    positions = np.full((sampler.num_photons, 3), world.source_position)
    directions = photon_generator.sample_directions(sampler.randoms[:,0:2])
    energies = photon_generator.sample_energies(sampler.randoms[:,2])
    intensities = np.ones(sampler.num_photons, dtype=float)

    logging.info(f"Generated {sampler.num_photons} photons with positions {positions.shape}, directions {directions.shape}, and energies {energies.shape}.")

    new_positions, entry_points, crossed_voxels, crossed_materials, distances, \
        forward_iterations = world.traverse_grid_air(positions, directions)
    positions = new_positions.copy()
    
    for scatter_index in range(sampler.num_scatters):
        distance_randoms = sampler.randoms[:,(1+scatter_index)*3]
        compton_randoms = sampler.randoms[:,(1+scatter_index)*3+1:(1+scatter_index)*3+3]
        logging.info(f"Scatter iteration {scatter_index + 1}/{sampler.num_scatters}.")
        arrive_detector_positions, detector_signals, intensities, positions, energies \
            = world.fd_iteration(positions, directions, distance_randoms, compton_randoms, att, energies, intensities)
        
        if scatter_index == 0:
            primary_detector.add_signal(arrive_detector_positions, detector_signals)
        else:
            scatter_detector.add_signal(arrive_detector_positions, detector_signals)

    eval = Evaluation(
        primary_detector.detector_grid,
        scatter_detector.detector_grid)
    eval.store_detector_grids(phantom.phantom_type_path)
    
pass