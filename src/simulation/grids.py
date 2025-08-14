import numpy as np
import os
from ..utils.logger import setup_logger
from ..simulation.attenuation import Attenuation
from ..simulation.compton import ComptonScattering

class WorldGrid:
    def __init__(self, path: str, source_position: np.ndarray = None,
                 voxel_size: float = 1.0):
        """
        Args:
            path: Path to .npy file (3D grid: [Z,Y,X])
            allow_energy_axis: If True, allow slicing along energy axis
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Voxel grid file not found: {path}")
        
        self.grid = np.load(path)  # shape: (Z, Y, X) or (Z, Y, X, E)
        self.grid_shape = self.grid.shape
        self.voxel_size = voxel_size
        self.logger = setup_logger(self.__class__.__name__)

        if source_position is None:
            self.source_position = np.array([self.grid_shape[0] // 2, self.grid_shape[1] // 2, 0])
            self.logger.info(f"Using default source position: {self.source_position}")
        else:
            self.source_position = np.array(source_position)
            self.logger.info(f"Using given source position: {self.source_position} in shape {self.grid_shape}")

        self.logger.info(f"WorldGrid initialized with shape {self.grid_shape}, voxel size {self.voxel_size}, source position {self.source_position}.")
        self.materials = self.get_all_materials()

    def get_values(self, indices: np.ndarray) -> np.ndarray:
        materials = self.grid[indices[:, 0], indices[:, 1], indices[:, 2]]
        return materials.reshape(-1, 1)
    
    def traverse_grid(self, positions: np.ndarray, directions: np.ndarray, through_air: bool = True):
        num_photons, num_dims = positions.shape
        entry_points_slices = []
        crossed_voxels_slices = []
        crossed_materials_slices = []
        forward_iterations = np.zeros(num_photons, dtype=int)
        distances_slices = []
        
        # positions = positions / self.voxel_size
        exit_grid = np.full(num_photons, False)
        arrived = np.full(num_photons, False)
        active = ~(exit_grid | arrived)

        current_voxel_idx_active = np.floor(positions).astype(int)
        neg_dir_mask = (directions < 0)
        pos_dir_mask = (directions > 0)
        on_boundary_mask_active = np.isclose(positions, current_voxel_idx_active)
        
        next_voxel_idx_active = current_voxel_idx_active # removed .copy()
        next_voxel_idx_active[neg_dir_mask & on_boundary_mask_active] -= 1

        exit_grid[np.any(next_voxel_idx_active < 0, axis=1) | np.any(next_voxel_idx_active >= self.grid_shape, axis=1)] = True
        active = ~(exit_grid | arrived)
        next_voxel_idx_active = next_voxel_idx_active[active, :]
        current_voxel_idx_active = current_voxel_idx_active[active, :]

        next_materials = self.get_values(next_voxel_idx_active)
        arrived_active = ((next_materials != 0) if through_air else (next_materials == 0)).ravel()
        next_materials = next_materials[~arrived_active]
        current_voxel_idx_active = current_voxel_idx_active[~arrived_active, :]
        arrived[active] = arrived_active
        active = ~(exit_grid | arrived)
        # here
        next_voxel_idx_active = next_voxel_idx_active[~arrived_active,:]

        on_boundary_mask_active = np.isclose(positions[active, :], np.floor(positions[active, :]))
        neg_dir_mask_active = neg_dir_mask[active, :]
        pos_dir_mask_active = pos_dir_mask[active, :]

        while np.any(active):
            forward_iterations[active] += 1

            current_crossed_voxels = np.full((num_photons, num_dims), -1, dtype=int)
            current_crossed_voxels[active,:] = next_voxel_idx_active
            
            crossed_voxels_slices.append(current_crossed_voxels)

            current_materials = np.full((num_photons), -1, dtype=int)
            current_materials[active] = next_materials.reshape(-1)
            crossed_materials_slices.append(current_materials)

            current_entry_points = np.full((num_photons, num_dims), np.nan, dtype=float)
            current_entry_points[active, :] = positions[active, :]
            entry_points_slices.append(current_entry_points)

            current_distances = np.full((num_photons), 0, dtype=float)
            max_dist_active = np.full(directions[active, :].shape, np.inf, dtype=float)
            active_directions = directions[active, :]
            active_positions = positions[active, :]
            max_dist_active[neg_dir_mask_active & on_boundary_mask_active] = -1 / active_directions[neg_dir_mask_active & on_boundary_mask_active]
            max_dist_active[neg_dir_mask_active & ~on_boundary_mask_active] = -(active_positions[neg_dir_mask_active & ~on_boundary_mask_active] - np.floor(active_directions[neg_dir_mask_active & ~on_boundary_mask_active])) / active_directions[neg_dir_mask_active & ~on_boundary_mask_active]
            max_dist_active[pos_dir_mask_active] = (1 - (active_positions[pos_dir_mask_active] - np.floor(active_positions[pos_dir_mask_active]))) / active_directions[pos_dir_mask_active]
            distances_go = np.min(max_dist_active, axis=1)
            current_distances[active] = distances_go
            distances_slices.append(current_distances)

            positions[active, :] += directions[active, :] * distances_go[:, np.newaxis]
            current_voxel_idx_active = np.floor(positions[active, :]).astype(int)
            
            on_boundary_mask_active = np.isclose(positions[active, :], current_voxel_idx_active)
            neg_dir_mask_active = neg_dir_mask[active, :]
            pos_dir_mask_active = pos_dir_mask[active, :]

            next_voxel_idx_active = current_voxel_idx_active.copy()
            next_voxel_idx_active[neg_dir_mask_active & on_boundary_mask_active] -= 1

            exid_grid_active = np.any(next_voxel_idx_active < 0, axis=1) | np.any(next_voxel_idx_active >= self.grid_shape, axis=1)
            pos_dir_mask_active = pos_dir_mask_active[~exid_grid_active]
            neg_dir_mask_active = neg_dir_mask_active[~exid_grid_active]
            on_boundary_mask_active = on_boundary_mask_active[~exid_grid_active]
            current_voxel_idx_active = current_voxel_idx_active[~exid_grid_active, :]
            next_voxel_idx_active = next_voxel_idx_active[~exid_grid_active, :]
            exit_grid[active] = exid_grid_active
            active = ~(exit_grid | arrived)
            
            next_materials = self.get_values(current_voxel_idx_active)
            arrived_active = ((next_materials != 0) if through_air else (next_materials == 0)).reshape(-1)
            pos_dir_mask_active = pos_dir_mask_active[~arrived_active]
            neg_dir_mask_active = neg_dir_mask_active[~arrived_active]
            on_boundary_mask_active = on_boundary_mask_active[~arrived_active]
            current_voxel_idx_active = current_voxel_idx_active[~arrived_active, :]
            next_voxel_idx_active = next_voxel_idx_active[~arrived_active, :]
            next_materials = next_materials[~arrived_active]
            arrived[active] = arrived_active
            active = ~(exit_grid | arrived)

            if np.sum(active) != 0 and np.max(forward_iterations) % 100 == 0:
                self.logger.info(f"Forward Iteration {np.max(forward_iterations)} completed, active photons: {np.sum(active)}")

        entry_points = np.stack(entry_points_slices, axis=0)
        crossed_voxels = np.stack(crossed_voxels_slices, axis=0)
        crossed_materials = np.stack(crossed_materials_slices, axis=0)
        distances = np.stack(distances_slices, axis=0)
        self.logger.info(f"Traversal completed. Total photons: {num_photons}, Active photons: {np.sum(active)}, Exit grid: {np.sum(exit_grid)}")
        return positions, entry_points, crossed_voxels, crossed_materials, distances, forward_iterations
    
    def traverse_grid_air(self, positions: np.ndarray, directions: np.ndarray):
        self.logger.info("Traversing through air.")
        return self.traverse_grid(positions, directions, through_air=True)
    
    def traverse_grid_material(self, positions: np.ndarray, directions: np.ndarray):
        self.logger.info("Traversing through phantom materials.")
        return self.traverse_grid(positions.copy(), directions, through_air=False)
    
    def get_all_materials(self) -> np.ndarray:
        return np.unique(self.grid)
    
    def fd_iteration(self, positions: np.ndarray, directions: np.ndarray, \
                     distance_randoms: np.ndarray, compton_randoms: np.ndarray, attenuation: Attenuation, \
                     energies: np.ndarray, initial_intensities: np.ndarray):
        exit_phantom_positions, entry_points, crossed_voxels, crossed_materials_phantom, distances, \
            forward_iterations = self.traverse_grid_material(positions, directions)
        crossed_voxels_active_mask = np.all(crossed_voxels >= 0, axis=2)
        # materials_map = np.full_like(crossed_voxels_active_mask, -1, dtype=int)
        # materials_map[crossed_voxels_active_mask] = self.get_values(crossed_voxels[crossed_voxels_active_mask]).reshape(-1)
        materials_map=crossed_materials_phantom
        energies_map = np.stack([energies] * materials_map.shape[0])
        energies_map[~crossed_voxels_active_mask] = np.nan
        attenuations_coeff_map = np.full_like(crossed_voxels_active_mask, 0, dtype=float)

        for material_index in range(len(self.materials)):
            material = self.materials[material_index]
            material_mask = (materials_map == material)
            if np.sum(material_mask) == 0:
                self.logger.info(f"No photons crossed material {attenuation.materials_str[material_index]}.")
            else:
                attenuations_coeff_map[material_mask] = attenuation.total_with_coherent_funs[material](energies_map[material_mask])*self.voxel_size

        # Calc full attenuation
        attenuations_in_voxels = attenuations_coeff_map*distances
        attenuations_through_material = np.sum(attenuations_in_voxels, axis=0)
        intensity_behind_material = np.exp(-attenuations_through_material)

        sampled_intensity = distance_randoms * (1-intensity_behind_material)
        att_int_goal_value = - np.log(1-sampled_intensity)
        passed_voxels_until_distance_hit = np.zeros_like(sampled_intensity, dtype=int)
        curr_distances_sums = np.zeros_like(sampled_intensity, dtype=float)
        curr_attenuation_sums = np.zeros_like(sampled_intensity, dtype=float)
        photons_active_mask = np.full_like(sampled_intensity, True, dtype=bool)
        voxel_index = 0
        while np.any(photons_active_mask):
            curr_voxel_distances = distances[voxel_index,photons_active_mask]
            curr_voxel_att_coeffs = attenuations_coeff_map[voxel_index,photons_active_mask]
            curr_voxel_att = curr_voxel_distances * curr_voxel_att_coeffs # HERE MAYBE MULTIPLY BY VOXEL SIZE?
            over_goal_mask = (curr_attenuation_sums[photons_active_mask]+curr_voxel_att) >= att_int_goal_value[photons_active_mask]
            if np.any(over_goal_mask):
                combined_mask = photons_active_mask.copy()
                combined_mask[photons_active_mask] = over_goal_mask
                rest_int = att_int_goal_value[combined_mask] - curr_attenuation_sums[combined_mask]
                rest_dist = np.zeros_like(rest_int, dtype=float)
                rest_dist[rest_int!=0] = rest_int[rest_int!=0]/curr_voxel_att_coeffs[over_goal_mask][rest_int!=0]  # HERE MAYBE DEVIDE BY VOXEL SIZE?
                curr_distances_sums[combined_mask] += rest_dist
                curr_attenuation_sums[combined_mask] += rest_dist * curr_voxel_att[over_goal_mask]
            photons_active_mask[photons_active_mask] = ~over_goal_mask
            curr_distances_sums[photons_active_mask] += curr_voxel_distances[~over_goal_mask]
            curr_attenuation_sums[photons_active_mask] += curr_voxel_att[~over_goal_mask]
            voxel_index += 1
            passed_voxels_until_distance_hit[photons_active_mask] += 1
        
        positions += directions * curr_distances_sums[:, np.newaxis]
        exit_phantom_intensities = np.exp(-attenuations_through_material) * initial_intensities
        scatter_intensities = (1-exit_phantom_intensities) * initial_intensities
        detector_signals = exit_phantom_intensities * energies
        
        arrive_detector_positions, entry_points, crossed_voxels, crossed_materials, distances, \
            forward_iterations = self.traverse_grid_air(exit_phantom_positions, directions)

        passed_voxels_until_distance_hit_i = np.arange(passed_voxels_until_distance_hit.shape[0])
        # ended_voxels = crossed_voxels[passed_voxels_until_distance_hit, passed_voxels_until_distance_hit_i, :]
        ended_materials = crossed_materials_phantom[passed_voxels_until_distance_hit, passed_voxels_until_distance_hit_i]
        compton_attenuation_coeffs = np.zeros_like(ended_materials, dtype=float)
        pass
        for material in self.materials[self.materials != 0]:
            material = self.materials[material_index]
            material_in_voxel_mask = (ended_materials == material)
            relevant_energies = energies[material_in_voxel_mask]
            total_fun_values = attenuation.total_without_coherent_funs[material](relevant_energies)
            compton_fun_values = attenuation.incoherent_funs[material](relevant_energies)
            compton_attenuation_coeffs[material_in_voxel_mask] = compton_fun_values
            scatter_intensities[material_in_voxel_mask] *= compton_fun_values / total_fun_values

        scatter = ComptonScattering(energies, directions, compton_randoms, compton_attenuation_coeffs)
        scatter.scatter()
        directions = scatter.scattered_directions
        energies = scatter.scattered_energies

        return arrive_detector_positions, detector_signals, scatter_intensities, positions, energies, directions
    