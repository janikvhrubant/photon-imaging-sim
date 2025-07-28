import numpy as np
from ..utils.logger import phantom_logger, setup_logger
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from pyvista.grid_objects import UniformGrid
import pyvista as pv

class WaterWall:
    def __init__(self, shape: tuple, wall_thickness: float = 5.0, 
                 voxel_size: float = 1.0, detector_distance: float = 0.0):
        """
        Initialize the VoxelWorld with a given shape, wall thickness, and voxel size.
        Args:
            shape (tuple): (x, y, z) dimensions of the grid (number of voxels).
            wall_thickness (float): Thickness of the water wall (in world units).
            voxel_size (float): Size of a single voxel (in world units).
        """
        self.shape = shape
        self.wall_thickness = float(wall_thickness)
        self.voxel_size = float(voxel_size)
        self.detector_distance = float(detector_distance)

        self.phantom_type_path = 'water_wall/{wall_thickness}mmThick__{detector_distance}mmDetectorDistance__{voxel_size}mmVoxelSize'.format(
            wall_thickness=str(self.wall_thickness).replace('.', '_'),
            detector_distance=str(self.detector_distance).replace('.', '_'),
            voxel_size=str(self.voxel_size).replace('.', '_')
        )

        self.phantom_path = './data/input/phantoms/{self.phantom_type_path}'
        if not os.path.exists(self.phantom_path):
            os.makedirs(self.phantom_path, exist_ok=True)
        self.description_log = phantom_logger(os.path.join(self.phantom_path, "description.txt"))
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(f"Initialized WaterWall with shape {self.shape}, wall thickness {self.wall_thickness}, voxel size {self.voxel_size} and detector distance {self.detector_distance}.")
        
        self.description_log.info(f"WaterWall shape: {shape}")
        self.description_log.info(f"WaterWall wall thickness: {self.wall_thickness} world units")
        self.description_log.info(f"WaterWall voxel size: {self.voxel_size} world units")
        self.description_log.info(f"WaterWall detector distance: {self.detector_distance} world units")

        self._generate_material_grid()
        self.description_log.info(f"Material grid generated and saved to {self.grid_path}")

    def _generate_material_grid(self):
        grid = np.zeros(self.shape, dtype=np.uint8)

        # Convert wall_thickness from world units to voxel units
        thickness_voxels = int(np.ceil(self.wall_thickness / self.voxel_size))
        self.description_log.info(f"WaterWall thickness in voxels: {thickness_voxels}")
        distance_voxels = int(np.ceil(self.detector_distance / self.voxel_size))
        self.description_log.info(f"WaterWall distance to detector in voxels: {distance_voxels}")

        grid[:, :, -distance_voxels-thickness_voxels:-distance_voxels] = 1
        self.logger.info(f"WaterWall with shape {grid.shape} generated with wall thickness {self.wall_thickness} world units and voxel size {self.voxel_size} world units.")

        self.grid = grid

        self.grid_path = os.path.join(self.phantom_path, "material_grid.npy")
        np.save(os.path.join(self.grid_path), self.grid)
        self.description_log.info(f"Material grid saved to {self.grid_path}")
        # self._save_projection_images()
        self._save_projection_images_pyvista()

    def _save_projection_images(self):

        grid = self.grid
        shape = grid.shape
        filled = np.ones_like(grid, dtype=bool)

        xs, ys, zs = np.where(filled)

        colors = np.empty(xs.shape[0], dtype=object)
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            val = grid[x, y, z]
            if val == 1:
                colors[i] = (0.0, 0.0, 1.0, 0.6)  # blue
            else:
                colors[i] = (0.6, 0.6, 0.6, 0.05)  # very light gray

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        voxel_size = 1
        for x, y, z, color in zip(xs, ys, zs, colors):
            ax.bar3d(x, y, z, voxel_size, voxel_size, voxel_size, color=color, shade=True, edgecolor='k', linewidth=0.1)

        ax.set_box_aspect(shape)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(self.phantom_path, 'voxel_bar3d_full.png'), dpi=300)
        plt.close(fig)

    def _save_projection_images_pyvista(self):
        # Make sure grid is 3D uint8
        data = np.asarray(self.grid, dtype=np.uint8)
        if data.ndim != 3:
            raise ValueError("self.grid must be a 3D numpy array")

        dims = data.shape
        voxel_size = getattr(self, 'voxel_size', 1.0)
        padded = np.pad(data, ((0, 1), (0, 1), (0, 1)), mode='constant', constant_values=0)

        grid = pv.ImageData()
        grid.dimensions = np.array(padded.shape)
        grid.origin = (0, 0, 0)
        grid.spacing = (voxel_size, voxel_size, voxel_size)
        grid.point_data["material"] = padded.flatten(order="F")

        opacity = [0.5, 0.6]         # air = transparent, water = visible
        cmap = ["#999999", "#0000ff"] # gray, blue

        os.makedirs(self.phantom_path, exist_ok=True)

        views = [
            ("front", (1, 0, 0), (0, 0, 1)),   # x+
            ("side", (0, 1, 0), (0, 0, 1)),    # y+
            ("top", (0, 0, 1), (0, 1, 0)),     # z+
            ("iso", (1, 1, 1), (0, 0, 1)),     # diagonal
            ("tilted", (1, 1, 0.5), (0, 0, 1)),
        ]

        # Compute the center and diagonal length of the volume
        size = np.array(padded.shape) * voxel_size
        center = size / 2
        diagonal = np.linalg.norm(size)

        for name, direction, viewup in views:
            p = pv.Plotter(off_screen=True)
            p.set_background("white")
            p.add_volume(grid, opacity=opacity, cmap=cmap)

            # Normalize direction vector
            direction = np.array(direction)
            direction = direction / np.linalg.norm(direction)

            # Position the camera far enough away to see the whole object
            camera_position = center + direction * diagonal * 2.0

            p.camera_position = [camera_position.tolist(), center.tolist(), viewup]
            p.camera.zoom(1.0)  # Optional fine-tuning

            path = os.path.join(self.phantom_path, f"pyvista_{name}.png")
            p.show(screenshot=path)
            p.close()
