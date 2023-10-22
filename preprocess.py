import numpy as np 
import open3d as o3d 
import trimesh
import plotly.graph_objects as go
import numpy as np

mesh = o3d.io.read_triangle_mesh('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/tr_reg_000.ply')

voxel_grid=o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,voxel_size=0.01)


voxels = voxel_grid.get_voxels()
indices = np.stack(list(vx.grid_index for vx in voxels))
colors = np.max(np.stack(list(vx.color for vx in voxels)),1)


ax = plt.figure().add_subplot(projection='3d')
ax.voxels(colors)

plt.show()


vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Bunny Visualize', width=800, height=600)
vis.add_geometry(voxel_grid)











d = np.load('/home/ubutnu/Documents/Projects/CorsoDeep/processed/tr_reg_000/vox_pc_64res_3000points.npz')

d['compressed_occupancies'].shape

T = trimesh.load_mesh('/home/ubutnu/Documents/Projects/CorsoDeep/processed/tr_reg_000/off_mesh.off')

v = np.load('/home/ubutnu/Documents/Projects/CorsoDeep/processed/tr_reg_001/vox_128.npy')

vol = v.T.reshape((64,64,64))

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(vol)

plt.show()



import plotly.graph_objects as go
import numpy as np
X, Y, Z = np.mgrid[0:1:64, 0:1:64, 0:1:64]
values = vol

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=v.flatten(),
    isomin=-0.0,
    isomax=255.0,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=50, # needs to be a large number for good volume rendering
    ))
fig.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA


import matplotlib.pyplot as plt
import numpy as np

dimensions = (32,32,32)

# start off all voxels as False (not drawn / transparent)
voxels = np.full(dimensions, True)

