from ..utils.logger import setup_logger
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class Attenuation:
    def __init__(self, materials: np.ndarray):
        self.logger = setup_logger(self.__class__.__name__)
        self.all_types = ['coherent', 'incoherent', 'photoelectric', 'total_with_coherent', 'total_without_coherent']
        self.coherent_funs = {}
        self.incoherent_funs = {}
        self.photoelectric_funs = {}
        self.total_with_coherent_funs = {}
        self.total_without_coherent_funs = {}
        material_string_df = pd.read_csv("data/input/attenuation/materials.csv")
        self.materials = materials
        self.materials_str = material_string_df.set_index('index').loc[materials, 'material'].values
        for i in range(len(materials)):
            material = materials[i]
            if material == 0:
                pass
            else:
                self.coherent_funs[material] = self._get_interpolation_function(material, 'coherent')
                self.incoherent_funs[material] = self._get_interpolation_function(material, 'incoherent')
                self.photoelectric_funs[material] = self._get_interpolation_function(material, 'photoelectric')
                self.total_with_coherent_funs[material] = self._get_interpolation_function(material, 'total_with_coherent')
                self.total_without_coherent_funs[material] = self._get_interpolation_function(material, 'total_without_coherent')
                self.logger.info(f"Attenuation initialized for {self.materials_str[i]}")

    def _get_interpolation_function(self, material: int, type: str):
        assert type in self.all_types, self.logger.error(f"Type {type} is not a valid attenuation type.")
        attenuation_data = pd.read_csv(f"data/input/attenuation/{material}.csv")
        energies = attenuation_data['photon_energy'].values*1e3
        values = attenuation_data[type].values
        return interp1d(energies, values, kind='linear', fill_value='extrapolate')
    