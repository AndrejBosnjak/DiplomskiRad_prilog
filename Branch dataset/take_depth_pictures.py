#Takes depth picutres with predetermined camera position stored in JSON files
#change_camera_view.py used to set camera positions

import numpy as np
import open3d as open3d
import os

#Asus camera
cameraWidth = 640
cameraHeight = 480

FX_RGB = 570.3422241210938
FY_RGB = 570.3422241210938
CX_RGB = 319.5
CY_RGB = 239.5

FX_DEPTH = 570.3422241210938
FY_DEPTH = 570.3422241210938
CX_DEPTH = 314.5
CY_DEPTH = 235.5

mesh = open3d.io.read_triangle_mesh("/home/andrej/anaconda3/envs/3bot/3d-model.stl") #model path
mesh.compute_vertex_normals()

vis = open3d.visualization.Visualizer()
vis.create_window(width=cameraWidth, height=cameraHeight)
vis.add_geometry(mesh)

view_control = vis.get_view_control()

folder_path = "CameraPositions_json"

i=0

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a JSON file
    if filename.endswith(".json"):
        # Construct the full path to the JSON file
        file_path = os.path.join(folder_path, filename)
        params = open3d.io.read_pinhole_camera_parameters(file_path) #camera params json file
        view_control.convert_from_pinhole_camera_parameters(params, True)
        vis.update_renderer()
        vis.capture_screen_image("ScreenImage" + str(i) + ".png", do_render=True)
        vis.capture_depth_image("DepthImage" + str(i) + ".png", do_render=True, depth_scale = 1)
        vis.capture_depth_point_cloud("PointCloud" + str(i) + ".ply", do_render=True)
        i=i+1
  

vis.run()

# Destroy the visualizer window
vis.destroy_window()