# X-Ray Photon Transport Simulation Framework

A modular and extensible simulation framework for **X-ray imaging** based on stochastic photon transport in voxelized phantoms.  
Supports both **Monte Carlo (MC)** and **Quasi-Monte Carlo (QMC)** sampling methods for comparative analysis of convergence behavior and image accuracy.

Developed as part of a Master's thesis in mathematics with a focus on variance reduction techniques in medical image simulation.

---

## 🔧 Features

- Photon transport using physical interaction models (Compton scattering, photoelectric absorption)
- Forced detection algorithm for improved variance reduction
- Modular sub-algorithms based on formal pseudocode (e.g., direction sampling, grid traversal)
- Support for voxelized phantoms and material-dependent attenuation coefficients
- MC, Sobol, and Halton sampling strategies for random number generation
- Spekpy-based X-ray tube spectrum integration
- Reproducible experiment configuration via JSON files
- Jupyter-based analysis notebooks for RMSE, convergence plots, and detector evaluation

---

## 🧪 Example Use Cases

- Comparing MC vs QMC for scatter estimation in X-ray imaging
- Generating simulated detector data for machine learning tasks
- Studying photon-material interaction patterns across complex geometries

---

## 🛠️ Requirements

- Python 3.10+
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Spekpy](https://spekpy.readthedocs.io/)

Install dependencies:
```bash
pip install -r requirements.txt
