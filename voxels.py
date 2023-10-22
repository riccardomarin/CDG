
import numpy as np
np.random.seed(1)
import torch
from torch.nn import Sequential, Linear, Flatten, Conv3d

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import tqdm
import torch.nn as nn
from torch.optim import SGD
import time
import sys 
import os
os.system('wget https://github.com/amitregmi/VoxNet-Google-Colab/raw/master/data/modelnet10.npz')

modelnet_file = 'modelnet10.npz'
data = np.load(modelnet_file, allow_pickle=True)

X, Y = shuffle(data['X_train'], data['y_train'])
X_test, Y_test = shuffle(data['X_test'], data['y_test'])

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

Y = to_categorical(Y, num_classes=10)

labels = {0 : 'bathtub',
          1 : 'bed',
          2 : 'chair',
          3 : 'desk',
          4 : 'dresser',
          5 : 'monitor',
          6 : 'nightstand',
          7 : 'sofa',
          8 : 'table',
          9 : 'toilet'}

model = Sequential(Conv3d(1, 64, 3), #input 30x30x30 -> 28 x 28 x 28
                   torch.nn.ReLU(),
                   Conv3d(64, 64, 5),  #input  -> 24 x 24 x 24
                   torch.nn.ReLU(),
                   Conv3d(64, 64, 10), #input  -> 15 x 15 x 15
                   torch.nn.ReLU(),
                   Flatten(),
                   Linear(15 * 15 *15 * 64,10),
                   torch.nn.Softmax()
                   ).cuda()
model.cuda()

################ Training
loss_history = []

loss = nn.MSELoss() #define the loss function
opt = SGD(model.parameters() , lr = 0.001) #define the gradient descent with learning rate as 0.001

start = time.time()

for k in range(10):
  print(k)
  for x,y in tqdm.tqdm(zip(X,Y)):
    x = torch.unsqueeze(torch.unsqueeze(torch.tensor(x,dtype=torch.float32),0),0).cuda()
    y = torch.unsqueeze(torch.unsqueeze(torch.tensor(y,dtype=torch.float32),0),0).cuda()
    opt.zero_grad() #to flush out the previous gradients
    loss_value = loss(model(x), y) #define the loss or error from model prediction and the actual label
    loss_value.backward() #calculate the gradients
    opt.step() #update the weights
    loss_history.append(loss_value)
    end = time.time()


torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss_value,
            }, 'chkpoint.pth')
################ Evaluation


checkpoint = torch.load('chkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

# Confusion Matrix
X_test_torch = torch.unsqueeze(torch.tensor(X_test,dtype=torch.float32),1).cuda()
Y_test_pred = torch.zeros(len(X_test_torch))

# Batch Evaluation
for i in np.arange(0,len(Y_test_pred)/10):
  Y_test_pred[int(i*10):int((i+1)*10)] = torch.argmax(model(X_test_torch[int(i*10):int((i+1)*10)]), axis=1)

# Confusion Matrix
conf = confusion_matrix(Y_test, Y_test_pred)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
plt.xticks(np.arange(0, 10, 1.0))
plt.yticks(np.arange(0, 10, 1.0))
ax.set_xticklabels(list(labels.values()))
ax.set_yticklabels(list(labels.values()))
plt.show()

avg_per_class_acc = np.mean(np.diagonal(conf) / np.sum(conf, axis=1))
print('Confusion matrix:\n{}'.format(conf))
print('Average per-class accuracy: {:.3f}'.format(avg_per_class_acc))


#######
import open3d as o3d 
from typing import Tuple
from sklearn.neighbors import KDTree
import trimesh
import utils.implicit_waterproofing as iw
import numpy as np 
import matplotlib.pyplot as plt



def voxelize(pc, res, bounds=(-1., 1.), save_path=None):
    grid_points = iw.create_grid_points_from_bounds(bounds[0], bounds[1], res)
    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    kdtree = KDTree(grid_points)
    _, idx = kdtree.query(pc)
    occupancies[idx] = 1

    if save_path is not None:
        compressed_occupancies = np.packbits(occupancies)
        if not exists(save_path):
            os.makedirs(save_path)
        np.savez(save_path, point_cloud=pc, compressed_occupancies=compressed_occupancies, bb_min=bounds[0],
                 bb_max=bounds[1], res=res)

    return occupancies

import glob
files = glob.glob('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/*.ply')

voxels_list = []
for f in files:
  o = o3d.io.read_point_cloud('/home/ubutnu/Documents/Projects/CorsoDeep/faust_ply/tr_reg_009.ply')
  voxels = voxelize(np.asarray(o.points), 30)
  voxels = voxels.reshape((30,30,30))  
  voxels_list.append(voxels) 
  
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxels)
plt.show()  
  
  
  
  
vsize=max(o.get_max_bound()-o.get_min_bound()) * 0.005
vsize=round(vsize,4)
        
voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=vsize)
bounds=voxel_grid.get_max_bound()-voxel_grid.get_min_bound()

  
voxels=voxel_grid.get_voxels()
vox_mesh=o3d.geometry.TriangleMesh()
for v in voxels:
    cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    cube.paint_uniform_color(v.color)
    cube.translate(v.grid_index, relative=False)
    vox_mesh+=cube

o3d.visualization.draw([vox_mesh])


voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o,
                                                            voxel_size=0.05)
o3d.visualization.draw_geometries([voxel_grid])

import matplotlib.pyplot as plt
import numpy as np

# prepare some coordinates
x, y, z = np.indices((30, 30, 30))


# combine the objects into a single boolean array

# 0 : bathtub
# 1 : Bed
# 2 : chair
# 3 : desk (?)
# 4:  dresser (?)
# 5 : monitor
# 6 : Nighstand (?)
# 7 : Sofas
# 8 : Tables
# 9 : Toilet

idx = 28
voxelarray = X_test[idx]
print(Y_test[idx])
# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray, edgecolor='k')

plt.show()