import open3d as o3d 
import os
import numpy as np 

v3f = o3d.utility.Vector3dVector
v3i = o3d.utility.Vector3iVector


os.system('wget https://github.com/czh-98/REALY/raw/master/data/3DDFA_v2.obj')

M = o3d.io.read_triangle_mesh('3DDFA_v2.obj')
M = M.simplify_quadric_decimation(target_number_of_triangles=1700)


o3d.visualization.draw([M])

tri = np.asarray(M.triangles)
vertices = np.asarray(M.vertices)


edges = np.vstack(
                    (tri[:,[0,1]], tri[:,[1,2]], tri[:,[2,0]],
                    tri[:,[1,0]], tri[:,[2,1]], tri[:,[0,2]])
                    )

unique_rows, counters = np.unique(edges, axis=0,  return_counts=True)


selected = unique_rows[counters == 1]
selected_vertices = np.squeeze(np.reshape(selected,(-1,1)))

colors = np.zeros((vertices.shape[0],3))

colors[selected_vertices,:] = [1,1,1]

N = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[selected_vertices]))

M.vertex_colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw([M,N])


o3d.io.load_triangle_mesh('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/tr_reg_000.ply')

flat_vertices = np.vstack((vertices[:,0],vertices[:,1],np.zeros(vertices[:,0].shape[0]))).T

M_flat = o3d.geometry.TriangleMesh(v3f(flat_vertices),M.triangles)


o3d.visualization.draw([M_flat])


import torch 
import scipy.sparse.linalg
from scipy.sparse import csr_matrix


M_flat_2D = flat_vertices[:,0:2]


u = np.zeros((flat_vertices.shape[0],))
u[selected_vertices] = M_flat_2D[selected_vertices,0]


v = np.zeros((flat_vertices.shape[0],))
v[selected_vertices] = M_flat_2D[selected_vertices,1]



L = np.zeros((flat_vertices.shape[0], flat_vertices.shape[0],))
L = csr_matrix(L)
edges1=tri[:,[0]]; edges2=tri[:,[1]];edges3=tri[:,[2]];

w = 10

L[edges1,edges2] = w
L[edges2,edges1] = w

L[edges2,edges3] = w
L[edges3,edges2] = w

L[edges3,edges1] = w
L[edges1,edges3] = w


L = - L / np.sum(L,axis=1)

L[selected_vertices,:] = 0


L = L + np.diag(np.ones(flat_vertices.shape[0]))





import scipy
S = scipy.sparse.linalg.spsolve(L,u)
T = scipy.sparse.linalg.spsolve(L,v)


uv = np.vstack((S,T,np.zeros(S.shape))).T
M_flat2 = o3d.geometry.TriangleMesh(v3f(uv),M.triangles)


import matplotlib.pyplot as plt


plt.subplot(121)
plt.scatter(uv[:,0],uv[:,1])
plt.subplot(122)
plt.scatter(flat_vertices[:,0],flat_vertices[:,1])
plt.show()


o3d.visualization.draw([M_flat2])

################



edges = np.vstack(
                    (tri[:,[0,1]], tri[:,[1,2]], tri[:,[2,0]],
                    tri[:,[1,0]], tri[:,[2,1]], tri[:,[0,2]])
                    )

unique_rows, counters = np.unique(edges, axis=0,  return_counts=True)





############
import open3d as o3d 
import os
import numpy as np 
import matplotlib.pyplot as plt

# Load mesh and convert to open3d.t.geometry.TriangleMesh
armadillo_data = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo_data.path)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

query_point = o3d.core.Tensor([[10, 10, 10]], dtype=o3d.core.Dtype.Float32)

# Compute distance of the query point from the surface
unsigned_distance = scene.compute_distance(query_point)
signed_distance = scene.compute_signed_distance(query_point)
occupancy = scene.compute_occupancy(query_point)

print("unsigned distance", unsigned_distance.numpy())
print("signed_distance", signed_distance.numpy())
print("occupancy", occupancy.numpy())

min_bound = mesh.vertex.positions.min(0).numpy()
max_bound = mesh.vertex.positions.max(0).numpy()

N = 256
query_points = np.random.uniform(low=min_bound, high=max_bound,
                                 size=[N, 3]).astype(np.float32)

# Compute the signed distance for N random points
signed_distance = scene.compute_signed_distance(query_points)

xyz_range = np.linspace(min_bound, max_bound, num=32)

# query_points is a [32,32,32,3] array ..
query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)

# signed distance is a [32,32,32] array
signed_distance = scene.compute_signed_distance(query_points)

# We can visualize a slice of the distance field directly with matplotlib
plt.imshow(signed_distance.numpy()[:, :, 15])
plt.show()


#########




from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from scipy.spatial.distance import cdist
from scipy.sparse import csc_matrix
from scipy import sparse
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.io import loadmat 



M = o3d.io.read_triangle_mesh('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/tr_reg_000.ply')
N = o3d.io.read_triangle_mesh('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/tr_reg_002.ply')


def dist_matrix(v, a):
    dist=cdist(v,v)
    values = dist[np.nonzero(a)]
    matrix = sparse.coo_matrix((values,np.nonzero(a)),shape=(v.shape[0],v.shape[0]))
    d = dijkstra(matrix,directed = False)
    return matrix ,d, dist


M_vertices = np.asarray(M.vertices)
N_vertices = np.asarray(N.vertices)
M_faces    = np.asarray(M.triangles)
N_faces    = np.asarray(N.triangles)


L = np.zeros((M_vertices.shape[0], M_vertices.shape[0],))
L = csr_matrix(L)
edges1=M_faces[:,[0]]; edges2=M_faces[:,[1]];edges3=M_faces[:,[2]];
w = 1
L[edges1,edges2] = w
L[edges2,edges1] = w

L[edges2,edges3] = w
L[edges3,edges2] = w

L[edges3,edges1] = w
L[edges1,edges3] = w



m_src, d_src, euc_dist_src = dist_matrix(M_vertices,L)
diama = np.max(d_src)

m_tar, d_tar, euc_dist_tar = dist_matrix(N_vertices,L)
diamb = np.max(d_tar)



M2 = o3d.io.read_triangle_mesh('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/tr_reg_020.ply')
N2 = o3d.io.read_triangle_mesh('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/tr_reg_022.ply')



M2_vertices = np.asarray(M2.vertices)
N2_vertices = np.asarray(N2.vertices)
M2_faces    = np.asarray(M2.triangles)
N2_faces    = np.asarray(N2.triangles)

m2_src, d2_src, euc2_dist_src = dist_matrix(M2_vertices,L)
diam2a = np.max(d2_src)

m2_tar, d2_tar, euc2_dist_tar = dist_matrix(N2_vertices,L)
diam2b = np.max(d2_tar)



np.max(euc_dist_src)
np.max(euc_dist_tar)

np.max(euc2_dist_src)
np.max(euc2_dist_tar)


o3d.visualization.draw([M2,N2])



edges = np.vstack(
                    (M_faces[:,[0,1]], M_faces[:,[1,2]], M_faces[:,[2,0]],
                    M_faces[:,[1,0]], M_faces[:,[2,1]], M_faces[:,[0,2]])
                    )


unique_rows, counters = np.unique(edges, axis=0,  return_counts=True)



m, d = dist_matrix(M_vertices,L)
diam = np.max(d)

idx = 1000
P = o3d.geometry.PointCloud(v3f(M_vertices[idx:idx+1,:]))

dz = d[:,1000]
norm = plt.Normalize()
colors = plt.cm.jet(norm(dz))
lower = dz.min()
upper = dz.max()
colors = plt.cm.jet((dz-lower)/(upper-lower))[:,0:3]
M.vertex_colors = v3f(colors)


dz = np.sum(d,axis=0)
norm = plt.Normalize()
colors = plt.cm.jet(norm(dz))
lower = dz.min()
upper = dz.max()
colors = plt.cm.jet((dz-lower)/(upper-lower))[:,0:3]
M.vertex_colors = v3f(colors)


o3d.visualization.draw([M])

m2, d2 = dist_matrix(N_vertices,L)
diam2 = np.max(d2)


U, S, Vh = np.linalg.svd(d)
S1_geod = S 


U, S, Vh = np.linalg.svd(d2)
S2_geod = S 


import matplotlib.pyplot as plt 
plt.plot(S1_geod[10:100])
plt.plot(S2_geod[10:100])
plt.show()



from scipy.spatial import distance_matrix
d = distance_matrix(M_vertices,M_vertices)
diam = np.max(d)

U, S, Vh = np.linalg.svd(d)
S1_euc = S 


d2 = distance_matrix(N_vertices,N_vertices)
diam2 = np.max(d2)

U, S, Vh = np.linalg.svd(d2)
S2_euc = S 

np.sum(np.abs(S1_geod-S2_geod))
np.sum(np.abs(S1_euc-S2_euc))


ax = plt.subplot(121)
ax.plot(S1_geod[100:6000])
ax.plot(S2_geod[100:6000])
ax.set_yscale('log')



ax1 = plt.subplot(122)
ax1.subplot(122)
ax1.plot(S1_euc[100:6000])
ax1.plot(S2_euc[100:6000])
ax1.set_yscale('log')

plt.show()


O = np.linalg.solve(L,u)


L[L<-1] = -1


O = np.linalg.solve(L,u)


####

import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from torch.utils.data import Dataset 

import glob 
from torch.utils.data import TensorDataset, DataLoader

lista = glob.glob('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/*.ply')
lista.sort()
l_v = []
ids = torch.tensor(np.repeat(np.arange(10),10))
for l in lista:
    M = o3d.io.read_triangle_mesh(l)
    v = np.asarray(M.vertices)
    l_v.append(v[np.newaxis])



ids_onehot = torch.tensor(F.one_hot(ids),dtype=torch.float32)


all_v = np.vstack(l_v)
my_dataset = TensorDataset(torch.tensor(all_v,dtype=torch.float32), torch.tensor(ids_onehot,dtype=torch.float32))
my_dataloader = DataLoader(my_dataset, batch_size = 4, shuffle=True)




class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        self.inputfc = nn.Linear(input_dim, 512)
        self.hidden = nn.Linear(512, 128)
        self.out = nn.Linear(128, output_dim)
        

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.inputfc(x))
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


model = Model(6890*3,10)

optimizer = optim.Adam(model.parameters())

losses = []
for e in np.arange(1000):
    for data,label in my_dataloader:
        preds = model(data)
        preds = torch.softmax(preds, 1)
        loss = F.cross_entropy(preds, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach().cpu().numpy())  
    
preds = model(data)

preds = torch.softmax(preds, 1)
torch.argmax(preds,axis=1)
torch.argmax(label,axis=1)


########