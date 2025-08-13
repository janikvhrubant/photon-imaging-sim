from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

import numpy as np
from ..utils.logger import phantom_logger, setup_logger

try:
    import pyvista as pv
    _HAS_PV = True
except Exception:
    _HAS_PV = False


class AbstractVoxelPhantom(ABC):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        voxel_size: float = 1.0,
        *,
        out_root: str = "./data/input/phantoms",
        metadata: Optional[Dict[str, Any]] = None,
        render_previews: bool = False,
    ) -> None:
        self.shape: Tuple[int, int, int] = tuple(int(x) for x in shape)
        if len(self.shape) != 3 or any(s <= 0 for s in self.shape):
            raise ValueError(f"Invalid shape {shape!r}. Must be positive 3-tuple.")

        self.voxel_size: float = float(voxel_size)
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be > 0.")

        self.out_root = out_root
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.metadata.setdefault("voxel_size", self.voxel_size)
        self.phantom_type_path = self._make_type_path_from_metadata(self.name, metadata)

        self.phantom_path = os.path.join(self.out_root, self.phantom_type_path)
        os.makedirs(self.phantom_path, exist_ok=True)

        self.description_log = phantom_logger(os.path.join(self.phantom_path, "description.txt"))
        self.logger = setup_logger(self.__class__.__name__)

        self.logger.info(
            f"Initialized {self.name} with shape={self.shape}, "
            f"voxel_size={self.voxel_size}."
        )
        if self.metadata:
            self.logger.info(f"Metadata: {self.metadata}")

        raw_grid = self._generate_material_grid()
        self.grid = self._validate_and_cast_grid(raw_grid)

        self.grid_path = os.path.join(self.phantom_path, "material_grid.npy")
        np.save(self.grid_path, self.grid)
        self.description_log.info(f"Material grid saved to {self.grid_path}")

        if render_previews:
            if not _HAS_PV:
                self.logger.warning("PyVista not available; skipping preview rendering.")
            else:
                try:
                    self._save_projection_images_pyvista()
                except Exception as exc:
                    self.logger.exception(f"Preview rendering failed: {exc}")

    @property
    def name(self) -> str:
        return self.NAME
    
    @abstractmethod
    def _generate_material_grid(self) -> np.ndarray:
        raise NotImplementedError

    def world_to_vox(self, length: float) -> int:
        return int(np.ceil(float(length) / self.voxel_size))

    def _validate_and_cast_grid(self, grid: Any) -> np.ndarray:
        arr = np.asarray(grid, dtype=np.uint8)
        if arr.ndim != 3:
            raise ValueError(f"Grid must be 3D, got ndim={arr.ndim}.")
        if tuple(arr.shape) != self.shape:
            raise ValueError(f"Grid shape {arr.shape} != expected {self.shape}.")
        return arr

    def _make_type_path(self) -> str:
        def _fmt(v: Any) -> str:
            s = str(v)
            return s.replace(".", "_").replace(" ", "")

        meta = "__".join(f"{k}={_fmt(v)}" for k, v in sorted(self.metadata.items()))
        base = self.name.strip().lower()
        return f"{base}/{meta}" if meta else base

    def _save_projection_images_pyvista(self) -> None:
        if not _HAS_PV:
            return

        data = np.asarray(self.grid, dtype=np.uint8)
        padded = np.pad(data, ((0, 1), (0, 1), (0, 1)), mode="constant", constant_values=0)

        grid = pv.ImageData()
        grid.dimensions = np.array(padded.shape)
        grid.origin = (0.0, 0.0, 0.0)
        grid.spacing = (self.voxel_size, self.voxel_size, self.voxel_size)
        grid.point_data["material"] = padded.flatten(order="F")

        opacity = [0.0, 0.6]
        cmap = ["#999999", "#1f77b4"]

        views = [
            ("front", (1, 0, 0), (0, 0, 1)),
            ("side",  (0, 1, 0), (0, 0, 1)),
            ("top",   (0, 0, 1), (0, 1, 0)),
            ("iso",   (1, 1, 1), (0, 0, 1)),
            ("tilted",(1, 1, 0.5), (0, 0, 1)),
        ]

        size = np.array(padded.shape, dtype=float) * float(self.voxel_size)
        center = size / 2.0
        diagonal = float(np.linalg.norm(size))

        for name, direction, viewup in views:
            p = pv.Plotter(off_screen=True)
            p.set_background("white")
            p.add_volume(grid, opacity=opacity, cmap=cmap)

            direction = np.array(direction, dtype=float)
            direction /= np.linalg.norm(direction)
            camera_position = center + direction * diagonal * 2.0

            p.camera_position = [camera_position.tolist(), center.tolist(), viewup]
            p.camera.zoom(1.0)

            path = os.path.join(self.phantom_path, f"pyvista_{name}.png")
            p.show(screenshot=path)
            p.close()

    @staticmethod
    def _make_type_path_from_metadata(name: str, metadata: Dict[str, Any]) -> str:
        def _fmt(v: Any) -> str:
            return str(v).replace(".", "_").replace(" ", "")
        meta = "__".join(f"{k}={_fmt(v)}" for k, v in sorted(metadata.items()))
        base = name.strip().lower()
        return f"{base}/{meta}" if meta else base
