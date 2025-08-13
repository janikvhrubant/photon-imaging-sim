import numpy as np
from sklearn.mixture import GaussianMixture

# --- Load data ---
file_path = 'data/output/detector_grids/mc1e5'
file_path = 'data/output/detector_grids/water_bone_wall/detector_distance_mm=10_0__n_bones=3__thickness_mm=20_0__voxel_size=1_0'
file_path = 'data/output/detector_grids/15mmQMC1e5'
file_path = 'data/output/detector_grids/BoneWaterWall/thick20.0mm/1E+05QMC15scatters/scatter_energies.npy'
file_path = 'data/output/detector_grids/BoneWaterWall/thick10.0mm/1E+05MC15scatters/scatter_energies.npy'

image = np.load(file_path, allow_pickle=True).astype(float)

# --- Mask invalids (NaN/Inf) ---
mask = np.isfinite(image)
x = image[mask].reshape(-1, 1)
if x.size == 0:
    raise ValueError("Keine finiten Werte in der Eingabe.")

# --- GMM-Clustering (K=3) ---
gmm = GaussianMixture(
    n_components=3,
    covariance_type="diag",  # 1D reicht 'diag'; in >1D ggf. 'full'
    reg_covar=1e-6,          # Stabilisierung gegen degenerierte Varianzen
    random_state=0,
    n_init=5,
    max_iter=500,
)
gmm.fit(x)

# Komponenten nach Mittelwert sortieren -> stabile Reihenfolge (klein/mittel/groß)
means_raw = gmm.means_.flatten()
order = np.argsort(means_raw)
label_remap = {old: new for new, old in enumerate(order)}

# Harte Zuordnung (MAP)
labels = gmm.predict(x)
labels_sorted = np.vectorize(label_remap.get)(labels)

# --- Mittelwert & Varianz pro Cluster (empirisch, ddof=1) ---
means = {}
variances = {}
for k in range(3):
    vals_k = x[labels_sorted == k, 0]
    if vals_k.size == 0:
        means[k] = np.nan
        variances[k] = np.nan
    elif vals_k.size == 1:
        means[k] = float(vals_k[0])
        variances[k] = np.nan  # Stichprobenvarianz nicht definiert für n=1
    else:
        means[k] = float(np.mean(vals_k))
        variances[k] = float(np.var(vals_k, ddof=1))

print("Mittelwerte pro Cluster:", means)
print("Varianzen pro Cluster:", variances)
