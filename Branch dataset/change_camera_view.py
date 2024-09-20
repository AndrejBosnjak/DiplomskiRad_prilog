#Used to set camera positions for take_depth_pictures.py

import numpy as np
import open3d as open3d

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


print("""-- Mouse view control --
  Left button + drag         : Rotate.
  Ctrl + left button + drag  : Translate.
  Wheel button + drag        : Translate.
  Shift + left button + drag : Roll.
  Wheel                      : Zoom in/out.

-- Keyboard view control --
  [/]          : Increase/decrease field of view.
  R            : Reset view point.
  Ctrl/Cmd + C : Copy current view status into the clipboard.
  Ctrl/Cmd + V : Paste view status from clipboard.

-- General control --
  Q, Esc       : Exit window.
  H            : Print help message.
  P, PrtScn    : Take a screen capture.
  D            : Take a depth capture.
  O            : Take a capture of current rendering settings.""")

mesh = open3d.io.read_triangle_mesh("/home/andrej/anaconda3/envs/3bot/3d-model.stl") #model path
mesh.compute_vertex_normals()

vis = open3d.visualization.Visualizer()
vis.create_window(width=cameraWidth, height=cameraHeight)
vis.add_geometry(mesh)

view_control = vis.get_view_control()
params = open3d.io.read_pinhole_camera_parameters("/home/andrej/anaconda3/envs/3bot/o3d_ASUS_camera_params_depth.json") #camera params json file
view_control.convert_from_pinhole_camera_parameters(params, True)

vis.run()

# Destroy the visualizer window
vis.destroy_window()


