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

    output_path = f"{config['PHANTOM']['object']}/thick{config['PHANTOM']['wall_thickness']}mm/{f"{float(config['SAMPLER']['num_photons']):.0E}".replace(".0E", "E")}{config['SAMPLER']['sampling_strategy']}{config['SAMPLER']['num_scatters']}scatters"

    phantom = PhantomSetup(config).phantom
    phantom.phantom_type_path = output_path
    
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

    worked_on_index = 0
    batch_size = int(float(config['SAMPLER'].get('batch_size', 1e5)))
    while worked_on_index < sampler.num_photons:
        curr_num_photons = min(batch_size, sampler.num_photons - worked_on_index)
        worked_on_end_index = worked_on_index + curr_num_photons
        logging.info("------------------------------------------------------------------")
        logging.info("------------------------------------------------------------------")
        logging.info(f"Processing photons from index {worked_on_index} to {worked_on_end_index}.")
        logging.info("------------------------------------------------------------------")
        logging.info("------------------------------------------------------------------")
        positions_batch = positions[worked_on_index:worked_on_end_index]
        directions_batch = directions[worked_on_index:worked_on_end_index]
        energies_batch = energies[worked_on_index:worked_on_end_index]
        intensities_batch = intensities[worked_on_index:worked_on_end_index]
        randoms_batch = sampler.randoms[worked_on_index:worked_on_end_index,:]

        new_positions, entry_points, crossed_voxels, crossed_materials, distances, \
            forward_iterations = world.traverse_grid_air(positions_batch, directions_batch)
        positions_batch = new_positions.copy()
        
        for scatter_index in range(sampler.num_scatters):
            distance_randoms = randoms_batch[:,(1+scatter_index)*3]
            compton_randoms = randoms_batch[:,(1+scatter_index)*3+1:(1+scatter_index)*3+3]
            logging.info(f"Scatter iteration {scatter_index + 1}/{sampler.num_scatters}.")
            arrive_detector_positions, detector_signals, intensities_batch, positions_batch, energies_batch \
                = world.fd_iteration(positions_batch, directions_batch, distance_randoms, compton_randoms, att, energies_batch, intensities_batch)
            
            if scatter_index == 0:
                primary_detector.add_signal(arrive_detector_positions, detector_signals)
            else:
                scatter_detector.add_signal(arrive_detector_positions, detector_signals)

        worked_on_index = worked_on_end_index

    eval = Evaluation(
        primary_energies=primary_detector.detector_grid,
        scatter_energies=scatter_detector.detector_grid)
    eval.store_detector_grids(phantom.phantom_type_path)
    eval.store_plots(phantom.phantom_type_path)
    
pass