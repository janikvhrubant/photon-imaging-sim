# main.py
from src.phantom.water_wall import WaterWall
from src.simulation.photon_generator import PhotonGenerator
import logging
import yaml
from src.simulation.grids import WorldGrid
import numpy as np
from src.utils.setup import SamplerSetup, PhantomSetup, PhotonGeneratorSetup
from src.simulation.attenuation import Attenuation

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
    materials = world.get_all_materials()
    att = Attenuation(materials)


    sampler = SamplerSetup(config).sampler

    photon_generator = PhotonGeneratorSetup(config).photon_generator
    photon_generator.update_angle(world)
    
    positions = np.full((sampler.num_photons, 3), world.source_position)
    directions = photon_generator.sample_directions(sampler.randoms[:,0:2])
    energies = photon_generator.sample_energies(sampler.randoms[:,2])

    logging.info(f"Generated {sampler.num_photons} photons with positions {positions.shape}, directions {directions.shape}, and energies {energies.shape}.")

    positions_in_material, entry_points, crossed_voxels, crossed_materials, distances, \
        forward_iterations = world.traverse_grid_air(positions, directions)
    positions = positions_in_material.copy()
    
    for i in range(sampler.num_scatters):
        distance_randoms = sampler.randoms[:,(1+i)*3]
        compton_randoms = sampler.randoms[:,(1+i)*3+1:(1+i)*3+3]
        positions_in_air, entry_points, crossed_voxels, crossed_materials, distances, \
            forward_iterations = world.traverse_grid_material(positions, directions)
        crossed_voxels_active_mask = mask = np.all(crossed_voxels >= 0, axis=2)
        materials_map = np.full_like(crossed_voxels_active_mask, -1, dtype=int)
        materials_map[crossed_voxels_active_mask] = world.get_values(crossed_voxels[crossed_voxels_active_mask]).reshape(-1)
        energies_map = np.stack([energies]*materials_map.shape[0])
        energies_map[~crossed_voxels_active_mask] = np.nan
        attenuations_coeff_map = np.full_like(crossed_voxels_active_mask, 0, dtype=float)

        for i in range(len(materials)):
            material = materials[i]
            material_mask = (materials_map == material)
            if np.sum(material_mask) == 0:
                logging.info(f"No photons crossed material {att.materials_str[i]}.")
            else:
                attenuations_coeff_map[material_mask] = att.total_with_coherent_funs[material](energies_map[material_mask])

        # Calc full attenuation
        attenuations_in_voxels = attenuations_coeff_map*distances
        attenuations_through_material = np.sum(attenuations_in_voxels, axis=0)
        intensity_behind_material = np.exp(-attenuations_through_material)

        sampled_intensity = distance_randoms * (1-intensity_behind_material)
        att_int_goal_value = - np.log(1-sampled_intensity)
        # index_until_distance_hit = np.zeros_like(sampled_intensity, dtype=int)
        distances_rnd_step = np.zeros_like(sampled_intensity, dtype=float)
        curr_attenuation_sums = np.zeros_like(sampled_intensity, dtype=float)
        photons_active_mask = np.full_like(sampled_intensity, True, dtype=bool)
        i = 0
        while np.any(photons_active_mask):
            if np.any(att_int_goal_value < curr_attenuation_sums):
                pass
            curr_voxel_distances = distances[i,photons_active_mask]
            curr_voxel_att_coeffs = attenuations_coeff_map[i,photons_active_mask]
            curr_voxel_att = curr_voxel_distances*curr_voxel_att_coeffs
            over_goal_mask = (curr_attenuation_sums[photons_active_mask]+curr_voxel_att) >= att_int_goal_value[photons_active_mask]
            if np.any(over_goal_mask):
                combined_mask = photons_active_mask.copy()
                combined_mask[photons_active_mask] = over_goal_mask
                rest_int = att_int_goal_value[combined_mask] - curr_attenuation_sums[combined_mask]
                rest_dist = rest_int/curr_voxel_distances[over_goal_mask]
                distances_rnd_step[combined_mask] += rest_dist
                curr_attenuation_sums[combined_mask] += rest_dist * curr_voxel_att[over_goal_mask]
                if np.any(att_int_goal_value < curr_attenuation_sums):
                    pass
                # photons_active_mask[combined_mask] = False
            photons_active_mask[photons_active_mask] = ~over_goal_mask
            distances_rnd_step[photons_active_mask] += curr_voxel_distances[~over_goal_mask]
            curr_attenuation_sums[photons_active_mask] += curr_voxel_att[~over_goal_mask]
            if np.any(att_int_goal_value < curr_attenuation_sums):
                pass
            i += 1
        positions += directions * distances_rnd_step[:, np.newaxis]
        pass


        # sampled_attenuations = 
        
        # attenuation


        # Calc attenuation_random position
