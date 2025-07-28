import numpy as np
import os
from src.utils.logger import setup_logger
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, primary_energies, scatter_energies):
        self.primary_energies = primary_energies
        self.scatter_energies = scatter_energies
        self.logger = setup_logger(self.__class__.__name__)

    @classmethod
    def from_file(self, file_path: str):
        primary_energies = np.load(file_path + '/primary_energies.npy')
        scatter_energies = np.load(file_path + '/scatter_energies.npy')
        return self(primary_energies, scatter_energies)

    def store_detector_grids(self, file_path: str):
        full_path = f'./data/output/detector_grids/{file_path}'
        # make sure full_path exists
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)

        np.save(os.path.join(full_path, 'primary_energies.npy'), self.primary_energies)
        np.save(os.path.join(full_path, 'scatter_energies.npy'), self.scatter_energies)
        np.save(os.path.join(full_path, 'full_energies.npy'), self.primary_energies + self.scatter_energies)

        self.logger.info(f"Detector grids stored at {full_path}.")

    def primary_plot(self, path: str):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.primary_energies, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Primary Energies')
        plt.title('Primary Energies Detector Grid')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(f'{path}/primary_energies.png')
        plt.close()
        self.logger.info("Primary energies plot saved.")

    def scatter_plot(self, path: str):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.scatter_energies, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Scatter Energies')
        plt.title('Scatter Energies Detector Grid')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(f'{path}/scatter_energies.png')
        plt.close()
        self.logger.info("Scatter energies plot saved.")

    def full_plot(self, path: str):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.primary_energies+self.scatter_energies, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Full Energies')
        plt.title('Full Energies Detector Grid')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(f'{path}/full_energies.png')
        plt.close()
        self.logger.info("Full energies plot saved.")
