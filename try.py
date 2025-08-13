from src.utils.evaluation import Evaluation

file_path = 'data/output/detector_grids/water_bone_wall/detector_distance_mm=10_0__n_bones=3__thickness_mm=5_0__voxel_size=1_0'
file_path = 'data/output/detector_grids/mc'
file_path = 'data/output/detector_grids/water_bone_wall/detector_distance_mm=10_0__n_bones=3__thickness_mm=20_0__voxel_size=1_0'
# file_path = 
eval = Evaluation.from_file(file_path)

eval.primary_plot(file_path)
eval.scatter_plot(file_path)
eval.full_plot(file_path)
     