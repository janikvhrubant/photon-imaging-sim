from src.phantom.water_wall import WaterWall

shape = (64, 64, 64)
wall_thickness = 5.0
voxel_size = 1.0
detector_distance = 10.0
assigned_density = {"water": 1.0}

phantom = WaterWall(
    shape=shape,
    wall_thickness=wall_thickness,
    voxel_size=voxel_size,
    detector_distance=detector_distance,
    assigned_density=assigned_density
)
phantom._generate_material_grid()
