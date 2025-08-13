from ..utils.logger import setup_logger
from ..utils.piecewise_log import _PiecewiseLogLog
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class Attenuation:
    def __init__(self, materials: np.ndarray):
        self.logger = setup_logger(self.__class__.__name__)
        self.all_types = ['coherent', 'incoherent', 'photoelectric', 'total_with_coherent', 'total_without_coherent']
        self.coherent_funs = {}
        self.incoherent_funs = {}
        self.photoelectric_funs = {}
        self.total_with_coherent_funs = {}
        self.total_without_coherent_funs = {}
        self.density_map = {1: 1.0, 2: 1.92}
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
        # self.plot_attenuation_functions()
        # pass

    def _get_interpolation_function(self, material: int, type: str):
        if type == 'incoherent':
            pass
        assert type in self.all_types, self.logger.error(f"Type {type} is not a valid attenuation type.")
        attenuation_data = pd.read_csv(f"data/input/attenuation/{material}.csv")
        energies = attenuation_data['photon_energy'].values*1e3
        print(energies)
        values = attenuation_data[type].values*self.density_map[material]
        func = interp1d(energies, values, kind='linear', fill_value='extrapolate')
        return func
    
    def plot_attenuation_functions(self):
        energies = np.arange(1, 100, .5)
        fig, axs = plt.subplots(
            len(self.materials) - 1, 
            len(self.all_types), 
            figsize=(15, 10), 
            sharex=True,   # x-Achse bleibt gleich
            sharey=False   # y-Achse individuell pro Subplot
        )

        for i, material in enumerate(self.materials):
            if material == 0:
                continue
            for j, type in enumerate(self.all_types):
                ax = axs[i-1, j]
                y_vals = self._get_interpolation_function(material, type)(energies)
                ax.plot(energies, y_vals, label=f"{self.materials_str[i]} - {type}")
                ax.set_title(f"{self.materials_str[i]} - {type}")
                ax.set_xlabel("Photon Energy (keV)")
                ax.set_ylabel("Attenuation Coefficient")
                ax.legend()

        plt.tight_layout()
        plt.show()

        pass
