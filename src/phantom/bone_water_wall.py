from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .base import AbstractVoxelPhantom  # adjust import


class WaterWallWithBones(AbstractVoxelPhantom):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        wall_thickness: float,
        n_bones: int,
        voxel_size: float = 1.0,
        *,
        detector_distance: float = 0.0,
        wall_material_id: int = 1,
        bone_material_id: int = 2,
        out_root: str = "./data/input/phantoms",
        metadata: Optional[Dict[str, Any]] = None,
        render_previews: bool = False,
    ) -> None:
        self.NAME = 'water_bone_wall'
        if n_bones <= 0:
            raise ValueError("n_bones must be a positive integer.")

        self.wall_thickness = float(wall_thickness)
        self.detector_distance = float(detector_distance)
        self.n_bones = int(n_bones)
        self.wall_material_id = int(wall_material_id)
        self.bone_material_id = int(bone_material_id)

        md = dict(metadata or {})
        md.setdefault("thickness_mm", self.wall_thickness)
        md.setdefault("detector_distance_mm", self.detector_distance)
        md.setdefault("n_bones", self.n_bones)
        md.setdefault("voxel_size", voxel_size)

        super().__init__(
            shape=shape,
            voxel_size=voxel_size,
            out_root=out_root,
            metadata=md,
            render_previews=render_previews,
        )

    def _generate_material_grid(self) -> np.ndarray:
        X, Y, Z = self.shape
        grid = np.zeros((X, Y, Z), dtype=np.uint8)

        t_vox = max(1, int(np.ceil(self.wall_thickness / self.voxel_size)))
        d_vox = max(0, int(np.ceil(self.detector_distance / self.voxel_size)))

        z_end = Z - d_vox
        if z_end <= 0:
            raise ValueError("detector_distance exceeds or equals grid depth.")
        z_start = max(0, z_end - t_vox)
        if z_start >= z_end:
            raise ValueError("wall_thickness too small after discretization.")

        grid[:, :, z_start:z_end] = self.wall_material_id

        r = X / (2.0 * self.n_bones)
        middle_y = Y / 2.0
        mid_points = [( (2*i + 1) * r, middle_y ) for i in range(self.n_bones)]

        xx, yy = np.meshgrid(np.arange(X, dtype=float),
                            np.arange(Y, dtype=float), indexing='ij')
        mask = np.zeros((X, Y), dtype=bool)
        r2 = r * r
        for cx, cy in mid_points:
            mask |= (xx - cx)**2 + (yy - cy)**2 <= r2

        grid[mask, z_start:z_end] = self.bone_material_id
        return grid

    def _make_type_path(self) -> str:
        def fmt(x): return str(x).replace(".", "_").replace(" ", "")
        return (
            "water_bone_wall/"
            f"thickness_mm={fmt(self.wall_thickness)}__"
            f"detector_mm={fmt(self.detector_distance)}__"
            f"voxel_size={fmt(self.voxel_size)}"
        )
