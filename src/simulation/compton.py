from src.utils.logger import setup_logger
import numpy as np

photon_rest_energy = 511

class ComptonScattering:
    def __init__(self, photon_energies, directions, randoms, attenuations):
        self.photon_energies = photon_energies
        self.directions = directions
        self.randoms_energy = randoms[:,0]
        self.randoms_direction = randoms[:,1]
        self.attenuations = attenuations
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("ComptonScattering initialized with photon energies, directions, randoms, and attenuations.")
        # self._satter

    def scatter(self):
        k = self.photon_energies/photon_rest_energy
        epsilon0_init = 1/(2*k+1)

        active_mask = np.full_like(self.photon_energies, True, dtype=bool)
        thetas = np.full_like(self.photon_energies, np.nan, dtype=float)

        while np.any(active_mask):
            epsilon0 = epsilon0_init[active_mask]
            r = np.random.uniform(size=np.sum(active_mask))
            epsilon = np.full_like(r, np.nan, dtype=float)
            r_small = (r < 0.5)
            if np.any(r_small):
                epsilon[r_small] = epsilon0[r_small] + (1-epsilon0[r_small])*r[r_small]
            if np.any(~r_small):
                epsilon[~r_small] = epsilon0[~r_small] + (1-epsilon0[~r_small])*2*(1-r[~r_small])

            cos_theta = 1 + 1/k[active_mask]*(1-1/epsilon)
            abs_cos_theta_small = (np.abs(cos_theta) <= 1)
            if np.any(abs_cos_theta_small):
                sin2_theta = 1 - cos_theta[abs_cos_theta_small]**2
                special_filter = (self.randoms_direction[active_mask] < (1/2) * (epsilon[abs_cos_theta_small] + 1/epsilon[abs_cos_theta_small] - sin2_theta))
                if np.any(special_filter):
                    combined_filter = abs_cos_theta_small.copy()
                    combined_filter[abs_cos_theta_small] = special_filter
                    combined_c_filter = active_mask.copy()
                    combined_c_filter[active_mask] = combined_filter.copy()
                    thetas[combined_c_filter] = np.arccos(cos_theta[combined_filter])
                    active_mask[combined_c_filter] = False
        
        self.angle = thetas
        self.scattered_energies = self.photon_energies / (1 + k*(1-np.cos(self.angle)))

        phis = 2 * np.pi * self.randoms_direction

        a = np.stack([np.array([0,0,1])]*len(self.photon_energies), axis=0)
        z_coord_not_one = (self.directions[:, 2] < .999)
        a[~z_coord_not_one] = np.array([1,0,0])
        u = np.cross(a, self.directions)/ np.linalg.norm(np.cross(a, self.directions), axis=1, keepdims=True)
        v = np.cross(self.directions, u)
        self.scattered_directions = (np.sin(self.angle)*np.cos(phis))[:, np.newaxis] * u + \
                                        (np.sin(self.angle)*np.sin(phis))[:, np.newaxis] * v + \
                                        (np.cos(self.angle))[:, np.newaxis] * self.directions
        
        self.logger.info(f"Computed angles and scattered directions for {len(self.photon_energies)} photons.")
