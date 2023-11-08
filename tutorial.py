## Load a 3D mesh

import open3d as o3d 
import os
import numpy as np 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import skimage

v3f = o3d.utility.Vector3dVector
v3i = o3d.utility.Vector3iVector

os.system('wget https://github.com/czh-98/REALY/raw/master/data/3DDFA_v2.obj')

M = o3d.io.read_triangle_mesh('3DDFA_v2.obj')

## Visualize a simple triangular mesh -- The simplest way
o3d.visualization.draw_plotly([M])

print(M)
print('Vertices:')
print(np.asarray(M.vertices))
print('Triangles:')
print(np.asarray(M.triangles))


## Visualize it -- Point cloud on Plotly
points = np.asarray(M.vertices)
colors = np.sin(np.asarray(M.vertices)/255 *10)

fig = go.Figure(
  data=[
    go.Scatter3d(
      x=points[:,0], y=points[:,1], z=points[:,2],
      mode='markers',
      marker=dict(size=1, color=colors)
)
],
  layout=dict(
    scene=dict(
      xaxis=dict(visible=False),
      yaxis=dict(visible=False),
      zaxis=dict(visible=False)
)
)
)
fig.show()


### Visualize it -- Fancy, multiplots, ...
# https://chart-studio.plotly.com/~empet/15416/mesh3d-with-vertex-intesities-and-face-i/#/

verts = np.asarray(M.vertices) #mymesh.points
faces = np.asarray(M.triangles) #mymesh.cells_dict['triangle']
x, y, z  = verts.T
i, j, k = faces.T

# subplots, three plot aligned in a row
fig = make_subplots(
          rows=1, cols=3,
          subplot_titles=('3d Mesh', 'Mesh3d with vertex intensities', 'Mesh3d with cell intensities'),
          horizontal_spacing=0.02,
          specs=[[{"type": "scene"}]*3])


tri_vertices = verts[faces]
Xe = []
Ye = []
Ze = []
for T in tri_vertices:
    Xe += [T[k%3][0] for k in range(4)] + [ None]
    Ye += [T[k%3][1] for k in range(4)] + [ None]
    Ze += [T[k%3][2] for k in range(4)] + [ None]

# Plot 1 - Mesh visualization
fig.add_trace(go.Scatter3d(x=Xe,
                     y=Ye,
                     z=Ze,
                     mode='lines',
                     name='',
                     line=dict(color= 'rgb(40,40,40)', width=0.5)), 1, 1);
# Lights
lighting = dict(ambient=0.5,
                diffuse=1,
                fresnel=4,
                specular=0.5,
                roughness=0.05,
                facenormalsepsilon=0)
lightposition=dict(x=100,
                   y=100,
                   z=10000)

# Plot 2 - Vertex color
fig.add_trace(go.Mesh3d(x=x, y=y, z=z,
                        i=i, j=j, k=k, colorscale='matter_r' ,
                        colorbar_len=0.85,
                        colorbar_x=0.625,
                        colorbar_thickness=20,
                        intensity=z, intensitymode='vertex',
                        flatshading=True), 1, 2);
fig.data[1].update(lighting=lighting);

# Plot 2 - Face color
fig.add_trace(go.Mesh3d(x=x, y=y, z=z,
                        i=i, j=j, k=k, colorscale='matter_r' ,
                        colorbar_len=0.85,
                        colorbar_x=0.97,
                        colorbar_thickness=20,
                        intensity=np.random.rand(len(faces)),
                        intensitymode='cell',
                        flatshading=True), 1, 3)
fig.data[2].update(lighting=lighting,
                   lightposition=lightposition)

# Camera and visualization layouts
fig.update_layout(width=1200, height=600, font_size=10)
fig.update_scenes(camera_eye_x=1.45, camera_eye_y=1.45, camera_eye_z=1.45);

fig.show()


#######
# Representation conversion 
# http://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html

M = o3d.io.read_triangle_mesh('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/tr_reg_000.ply')
vertices = np.asarray(M.vertices)
vertices = (vertices - np.min(vertices))/(np.max(vertices) - np.min(vertices))
M.vertices = o3d.utility.Vector3dVector(vertices)

mean_vertices = np.mean(vertices,axis=0)[np.newaxis]

mesh = o3d.t.geometry.TriangleMesh.from_legacy(M)
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)

query_point = o3d.core.Tensor(mean_vertices, dtype=o3d.core.Dtype.Float32)

unsigned_distance = scene.compute_distance(query_point)
signed_distance = scene.compute_signed_distance(query_point)
occupancy = scene.compute_occupancy(query_point)

print("unsigned distance", unsigned_distance.numpy())
print("signed_distance", signed_distance.numpy())
print("occupancy", occupancy.numpy())

####

# Set the grid
nc = 32

one = np.ones((nc, nc, nc))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(one, alpha = 0.12, edgecolor="k", shade=True) # Voxel visualization
ax.set_title('Voxels')
plt.tight_layout()
plt.show()

# Set the centers
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xc = np.arange(1, nc+1) - .5
yc = np.arange(1, nc+1) - .5
zc = np.arange(1, nc+1) - .5
xc_, yc_, zc_ = np.meshgrid(xc, yc, zc)
ax.scatter(xc_.ravel(), yc_.ravel(), zc_.ravel(), marker = 10, c = 'r', s = 100) # Voxel Centers
ax.voxels(one, alpha = 0.12, edgecolor="k", shade=True) # Voxels
plt.tight_layout()
ax.set_title('Voxels and Voxel centers')
plt.show()

# Query centers
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(one, alpha = 0.12, edgecolor="k", shade=True)
ax.scatter(vertices[:,0].ravel()*8,vertices[:,1].ravel()*8, vertices[:,2].ravel()*8, c = 'k')
ax.scatter(xc_.ravel(), yc_.ravel(), zc_.ravel(), marker = 10, c = 'r', s = 100)
ax.set_title('Random points in Voxel space')
plt.tight_layout()
plt.show()

##### Visualize Occupancy
query_coords = np.vstack((xc_.ravel(), yc_.ravel(), zc_.ravel())).T/nc
query_point = o3d.core.Tensor(query_coords, dtype=o3d.core.Dtype.Float32)

signed_distance = scene.compute_signed_distance(query_point)
signed_distance = signed_distance.numpy().reshape((nc,nc,nc))

occ = signed_distance < 0.01

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(occ, alpha = 0.12, edgecolor="k", shade=True)
ax.set_title('Occupancy')
plt.show()

##### From SDF to mesh
vertices, faces, normals, _ = skimage.measure.marching_cubes(signed_distance, level=0.01)
Mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
Mesh.compute_vertex_normals()
o3d.visualization.draw(Mesh)

##### Multi views
M = o3d.io.read_triangle_mesh('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/tr_reg_000.ply')
vertices = np.asarray(M.vertices)

mesh = o3d.t.geometry.TriangleMesh.from_legacy(M)

scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)

rays = scene.create_rays_pinhole(fov_deg=60,
                                 center=[-1.5,-1.5,-1.5],
                                 eye=[1,1,1],
                                 up=[0,0,1],
                                 width_px=1024,
                                 height_px=1024)

ans = scene.cast_rays(rays)

plt.imshow(ans['t_hit'].numpy())
plt.show()


######## UV Parametrization
import scipy
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from trimesh.visual import texture, TextureVisuals
from trimesh import Trimesh
from PIL import Image


# Load mesh
M_full = o3d.io.read_triangle_mesh('3DDFA_v2.obj')
M = M_full.simplify_quadric_decimation(100)

tri = np.asarray(M.triangles)
vertices = np.asarray(M.vertices)

# Identify the rim -> they have only one incident face
edges = np.vstack(
                    (tri[:,[0,1]], tri[:,[1,2]], tri[:,[2,0]],
                    tri[:,[1,0]], tri[:,[2,1]], tri[:,[0,2]])
                    )

unique_rows, counters = np.unique(edges, axis=0,  return_counts=True)

selected = unique_rows[counters == 1]
selected_vertices = np.squeeze(np.reshape(selected,(-1,1)))

colors = np.zeros((vertices.shape[0],3))
colors[selected_vertices,:] = [1,1,1]

#Visualize the rim
N = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[selected_vertices]))
M.vertex_colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw([M,N])

# Flatten the vertices 
flat_vertices = np.vstack((vertices[:,0],vertices[:,1],np.zeros(vertices[:,0].shape[0]))).T
M_flat = o3d.geometry.TriangleMesh(v3f(flat_vertices),M.triangles)
M_flat_2D = flat_vertices[:,0:2]

# Initialize the linear system
u = np.zeros((flat_vertices.shape[0],))
v = np.zeros((flat_vertices.shape[0],))

# For the rim vertices, we just set their U,V coordinates
u[selected_vertices] = M_flat_2D[selected_vertices,0]
v[selected_vertices] = M_flat_2D[selected_vertices,1]

# L is the constraints matrix
L = np.zeros((flat_vertices.shape[0], flat_vertices.shape[0],))
L = csr_matrix(L)

edges1=tri[:,[0]]; edges2=tri[:,[1]]; edges3=tri[:,[2]];

# If exists the edge, we add a constant weight
w = 10

L[edges1,edges2] = w
L[edges2,edges1] = w

L[edges2,edges3] = w
L[edges3,edges2] = w

L[edges3,edges1] = w
L[edges1,edges3] = w

print(L)

L = - L / np.sum(L,axis=1)
L[selected_vertices,:] = 0

L = L + np.diag(np.ones(flat_vertices.shape[0]))


# Solve the linear system
S = scipy.sparse.linalg.spsolve(L,u)
T = scipy.sparse.linalg.spsolve(L,v)

uv = np.vstack((S,T,np.zeros(S.shape))).T

# Visualize uv
plt.subplot(121)
plt.scatter(uv[:,0],uv[:,1])
plt.subplot(122)
plt.scatter(flat_vertices[:,0],flat_vertices[:,1])
plt.show()


# Map in 3D
M_flat2 = o3d.geometry.TriangleMesh(v3f(uv),M.triangles)
o3d.visualization.draw([M_flat2])


M = o3d.io.read_triangle_mesh('3DDFA_v2.obj')
img = Image.open('pattern.png') 

### Wrong UV
uv_wrong = flat_vertices[:,0:-1]
uv_wrong = (uv_wrong - np.min(uv_wrong))/(np.max(uv_wrong) - np.min(uv_wrong))

material = texture.SimpleMaterial(image=img)   
tex = TextureVisuals(uv=uv_wrong, image=img, material=material)

mesh = Trimesh(
            vertices=np.asarray(M.vertices),
            faces=np.asarray(M.triangles),
            visual=tex,
            validate=True,
            process=False
        )


mesh.export('test_wrong.obj')


#### Correct UV
uv = (uv - np.min(uv))/(np.max(uv) - np.min(uv))

material = texture.SimpleMaterial(image=img)   
tex = TextureVisuals(uv=uv, image=img, material=material)

mesh = Trimesh(
            vertices=np.asarray(M.vertices),
            faces=np.asarray(M.triangles),
            visual=tex,
            validate=True,
            process=False
        )


mesh.export('test.obj')

tri = np.asarray(M.triangles)

M.triangle_uvs = o3d.utility.Vector2dVector(uv[:,0:2][tri])

###### 







######


tree = cKDTree(np.c_[x.ravel(), y.ravel(), z.ravel()])

# query of the one (k=1) closest point to each voxel center
dd, ii = tree.query(np.c_[xc_.ravel(), yc_.ravel(), zc_.ravel()], k=1)

# to find corresponding values by indexing
x_close = x.ravel()[ii]
y_close = y.ravel()[ii]
z_close = z.ravel()[ii]


one[:] = signed_distance.reshape((nc,nc,nc))



plt.imshow(signed_distance[:, 70, :])
plt.colorbar()
plt.show()

xyz_range = np.linspace(0, 1, 1000)

# query_points is a [32,32,32,3] array ..
query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)



#####
query_coords = np.vstack((xc_.ravel(), yc_.ravel(), zc_.ravel())).T/nc
query_point = o3d.core.Tensor(query_coords, dtype=o3d.core.Dtype.Float32)

pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords))
o3d.visualization.draw([pc, M])

occ = []
for q in query_coords:
    qp = o3d.core.Tensor(q[np.newaxis], dtype=o3d.core.Dtype.Float32)
    unsigned_distance = scene.compute_distance(qp)
    signed_distance = scene.compute_signed_distance(qp)
    occupancy = scene.compute_occupancy(qp)
    occ.append(occupancy.numpy())
    
occupancy = np.hstack(occ)
one[:] = occupancy.reshape((nc,nc,nc))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(one, alpha = 0.12, edgecolor="k", shade=True)
ax.set_title('Random points in Voxel space')
plt.tight_layout()
plt.show()


x = vertices[:,0]*nc; y = vertices[:,1]*nc; z = vertices[:,2]*nc; 

from scipy.spatial import cKDTree
# create a KDTree datastructure for query

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(one, alpha = 0.12, edgecolor="k", shade=True)
ax.scatter(x.ravel(), y.ravel(), z.ravel(), c = 'k')
ax.scatter(xc_.ravel(), yc_.ravel(), zc_.ravel(), marker = 10, c = 'r', s = 100)

tree = cKDTree(np.c_[x.ravel(), y.ravel(), z.ravel()])
# query of the one (k=1) closest point to each voxel center
dd, ii = tree.query(np.c_[xc_.ravel(), yc_.ravel(), zc_.ravel()], k=1)

# to find corresponding values by indexing
x_close = x.ravel()[ii]
y_close = y.ravel()[ii]
z_close = z.ravel()[ii]

# visualization
for i in range(len(ii)):
    ax.plot3D([xc_.ravel()[i], x.ravel()[ii[i]]], [yc_.ravel()[i], y.ravel()[ii[i]]], [zc_.ravel()[i], z.ravel()[ii[i]]], c='g')
ax.set_title('Voxels, Voxel centers and Closest points')


plt.tight_layout()
plt.show()
#######



## Multi views

rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    fov_deg=90,
    center=[0, 0, 2],
    eye=[2, 3, 0],
    up=[0, 1, 0],
    width_px=640,
    height_px=480,
)
ans = scene.cast_rays(rays)

plt.imshow(ans['t_hit'].numpy())
plt.show()


pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_coords))
o3d.visualization.sce([pc,scene])

unsigned_distance = scene.compute_distance(query_point)
signed_distance = scene.compute_signed_distance(query_point)
occupancy = scene.compute_occupancy(query_point)

one[:] = occupancy.numpy().reshape((nc,nc,nc))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(one, alpha = 0.12, edgecolor="k", shade=True)
ax.set_title('Random points in Voxel space')
plt.tight_layout()
plt.show()




###




###


import matplotlib.pyplot as plt
import numpy as np


def midpoints(x):
    sl = ()
    for _ in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

# prepare some coordinates, and attach rgb values to each
r, g, b = np.indices((17, 17, 17)) / 16.0
rc = midpoints(r)
gc = midpoints(g)
bc = midpoints(b)

# define a sphere about [0.5, 0.5, 0.5]
sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2

# combine the color components
colors = np.zeros(sphere.shape + (3,))
colors[..., 0] = rc
colors[..., 1] = gc
colors[..., 2] = bc

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(r, g, b, sphere,
          facecolors=colors,
          edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
          linewidth=0.5)
ax.set(xlabel='r', ylabel='g', zlabel='b')
ax.set_aspect('equal')

plt.show()






# Create voxels
coords = np.linspace(0, 1,cells)
centers = coords[1:] - (coords[1:] - coords[:-1])

x,y,z = np.meshgrid(centers,centers,centers)



# Create axis
axes = [cells, cells, cells]

# Create Data
data = np.ones(axes, dtype=np.bool_)
 
# Control Transparency
alpha = 0.9
 
# Control colour
colors = np.empty(axes + [4], dtype=np.float32)
 
colors[:] = [1, 0, 0, alpha]  # red

edge = coords[1]-coords[0]

vox_x = np.int32(vertices[:,0] // edge)
vox_y = np.int32(vertices[:,1] // edge)
vox_z = np.int32(vertices[:,2] // edge)

occup = np.zeros((cells,cells,cells))
occup[vox_x,vox_y,vox_z]=True

# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(occup, facecolors=colors)


plt.show()




M = o3d.io.read_point_cloud('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/tr_reg_000.ply')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(M,voxel_size=0.1)

queries = np.asarray(M.points)
output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))

o3d.visualization.draw([voxel_grid])



queries = np.asarray(ply_file.points)
output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
print(output[:])