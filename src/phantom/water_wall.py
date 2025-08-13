from __future__ import annotations

from typing import Tuple, Optional, Dict, Any

import numpy as np

from .base import AbstractVoxelPhantom


class WaterWall(AbstractVoxelPhantom):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        wall_thickness: float,
        voxel_size: float = 1.0,
        *,
        detector_distance: float = 0.0,
        material_id: int = 1,
        out_root: str = "./data/input/phantoms",
        metadata: Optional[Dict[str, Any]] = None,
        render_previews: bool = True,
    ) -> None:
        self.NAME = 'water_wall'
        self.wall_thickness = float(wall_thickness)
        self.detector_distance = float(detector_distance)
        self.material_id = int(material_id)

        md = dict(metadata or {})
        md.setdefault("thickness_mm", self.wall_thickness)
        md.setdefault("detector_mm", self.detector_distance)
        md.setdefault("voxel_size", voxel_size)

        super().__init__(
            shape=shape,
            voxel_size=voxel_size,
            out_root=out_root,
            metadata=md,
            render_previews=render_previews,
        )

    def _generate_material_grid(self) -> np.ndarray:
        grid = np.zeros(self.shape, dtype=np.uint8)

        t_vox = int(np.ceil(self.wall_thickness / self.voxel_size))
        d_vox = int(np.ceil(self.detector_distance / self.voxel_size))

        t_vox = max(t_vox, 1)
        d_vox = max(d_vox, 0)

        z_end = self.shape[2] - d_vox
        if z_end <= 0:
            raise ValueError("detector_distance exceeds or equals grid depth.")
        z_start = max(0, z_end - t_vox)
        if z_start >= z_end:
            raise ValueError("wall_thickness too small after discretization.")

        grid[:, :, z_start:z_end] = self.material_id
        return grid

    def _make_type_path(self) -> str:
        def fmt(x): return str(x).replace(".", "_").replace(" ", "")
        return (
            "water_wall/"
            f"thickness_mm={fmt(self.wall_thickness)}__"
            f"detector_mm={fmt(self.detector_distance)}__"
            f"voxel_size={fmt(self.voxel_size)}"
        )
