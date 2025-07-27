import numpy as np
import os
from ..utils.logger import setup_logger

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
        
        positions = positions / self.voxel_size
        exit_grid = np.full(num_photons, False)
        arrived = np.full(num_photons, False)
        active = ~(exit_grid | arrived)

        current_voxel_idx = np.floor(positions).astype(int)
        neg_dir_mask = (directions < 0)
        pos_dir_mask = (directions > 0)
        on_boundary_mask_active = np.isclose(positions, current_voxel_idx)
        
        next_voxel_idx_active = current_voxel_idx.copy()
        next_voxel_idx_active[neg_dir_mask & on_boundary_mask_active] -= 1

        exit_grid[np.any(next_voxel_idx_active < 0, axis=1) & np.any(next_voxel_idx_active >= self.grid_shape, axis=1)] = True
        active = ~(exit_grid | arrived)
        next_voxel_idx_active = next_voxel_idx_active[active, :]
        current_voxel_idx = current_voxel_idx[active, :]

        if np.any(current_voxel_idx[:,2] >= 200):
            pass
        next_materials = self.get_values(current_voxel_idx)
        arrived_active = ((next_materials != 0) if through_air else (next_materials == 0)).ravel()
        next_materials = next_materials[~arrived_active]
        current_voxel_idx = current_voxel_idx[~arrived_active, :]
        arrived[active] = arrived_active
        active = ~(exit_grid | arrived)

        on_boundary_mask_active = np.isclose(positions[active, :], np.floor(positions[active, :]))
        neg_dir_mask_active = neg_dir_mask[active, :]
        pos_dir_mask_active = pos_dir_mask[active, :]

        if next_voxel_idx_active.shape[0]!=np.sum(active):
            pass

        while np.any(active):
            forward_iterations[active] += 1

            current_crossed_voxels = np.full((num_photons, num_dims), -1, dtype=int)
            current_crossed_voxels[active,:] = next_voxel_idx_active
            # TODO: fix ^
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
            current_voxel_idx = np.floor(positions[active, :]).astype(int)
            
            on_boundary_mask_active = np.isclose(positions[active, :], current_voxel_idx)
            neg_dir_mask_active = neg_dir_mask[active, :]
            pos_dir_mask_active = pos_dir_mask[active, :]

            next_voxel_idx_active = current_voxel_idx.copy()
            next_voxel_idx_active[neg_dir_mask_active & on_boundary_mask_active] -= 1

            exid_grid_active = np.any(next_voxel_idx_active < 0, axis=1) | np.any(next_voxel_idx_active >= self.grid_shape, axis=1)
            pos_dir_mask_active = pos_dir_mask_active[~exid_grid_active]
            neg_dir_mask_active = neg_dir_mask_active[~exid_grid_active]
            on_boundary_mask_active = on_boundary_mask_active[~exid_grid_active]
            current_voxel_idx = current_voxel_idx[~exid_grid_active, :]
            next_voxel_idx_active = next_voxel_idx_active[~exid_grid_active, :]
            exit_grid[active] = exid_grid_active
            active = ~(exit_grid | arrived)
            
            next_materials = self.get_values(current_voxel_idx)
            arrived_active = ((next_materials != 0) if through_air else (next_materials == 0)).reshape(-1)
            pos_dir_mask_active = pos_dir_mask_active[~arrived_active]
            neg_dir_mask_active = neg_dir_mask_active[~arrived_active]
            on_boundary_mask_active = on_boundary_mask_active[~arrived_active]
            current_voxel_idx = current_voxel_idx[~arrived_active, :]
            next_voxel_idx_active = next_voxel_idx_active[~arrived_active, :]
            next_materials = next_materials[~arrived_active]
            arrived[active] = arrived_active
            active = ~(exit_grid | arrived)
            if next_voxel_idx_active.shape[0]!=np.sum(active):
                pass

            if np.sum(active) != 0:
                self.logger.info(f"Forward Iteration {np.max(forward_iterations)} completed, active photons: {np.sum(active)}")

        entry_points = np.stack(entry_points_slices, axis=0)
        crossed_voxels = np.stack(crossed_voxels_slices, axis=0)
        crossed_materials = np.stack(crossed_materials_slices, axis=0)
        distances = np.stack(distances_slices, axis=0)
        self.logger.info(f"Traversal completed. Total photons: {num_photons}, Active photons: {np.sum(active)}, Exit grid: {np.sum(exit_grid)}")
        return positions, entry_points, crossed_voxels, crossed_materials, distances, forward_iterations
    
    def traverse_grid_air(self, positions: np.ndarray, directions: np.ndarray):
        return self.traverse_grid(positions, directions, through_air=True)
    
    def traverse_grid_material(self, positions: np.ndarray, directions: np.ndarray):
        return self.traverse_grid(positions, directions, through_air=False)
    
    def get_all_materials(self) -> np.ndarray:
        return np.unique(self.grid)
    