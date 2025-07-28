from src.utils.evaluation import Evaluation

file_path = './data/output/detector_grids/water_wall/10_0mmThick__10_0mmDetectorDistance__1_0mmVoxelSize'
eval = Evaluation.from_file(file_path)

eval.primary_plot(file_path)
eval.scatter_plot(file_path)
eval.full_plot(file_path)
