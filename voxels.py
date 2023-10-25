
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
import matplotlib.pyplot as plt 

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


####### VOXELIZATION
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
files.sort()

voxels_list = []
res = 64
for f in files:
  o = o3d.io.read_point_cloud(f)
  voxels = voxelize(np.asarray(o.points), res)
  voxels = voxels.reshape((res,res,res))  
  voxels_list.append(voxels[np.newaxis,:,:,:])
   
vox_faust = np.vstack(voxels_list)
np.save('vox_faust_' + str(res) +'.npy',vox_faust)
np.save('names_' + str(res) +'.npy',files)


############# FAUST 30 ###############

vox_faust = np.load('vox_faust_30.npy')
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(vox_faust[10])
plt.show()  
  

IDs   = np.int32(np.hstack((np.ones((10,))*0, np.ones((10,))*1, np.ones((10,))*2, np.ones((10,))*3, np.ones((10,))*4, 
                   np.ones((10,))*5, np.ones((10,))*6, np.ones((10,))*7, np.ones((10,))*8, np.ones((10,))*9)))
# IDs = to_categorical(IDs, num_classes=10)
poses = (np.repeat(np.arange(0,10)[np.newaxis],10,0)).flatten()



X, Y = shuffle(vox_faust, poses)
X_test, Y_test = shuffle(X[80:], Y[80:])
X = X[0:80]; Y = Y[0:80]


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

Y = to_categorical(Y, num_classes=10)

  
import random 
random_idxs = np.squeeze(random.sample(range(100),80))

# X_train      = X[random_idxs]
# IDs_train   = np.int32(IDs[random_idxs])
# IDs_train = to_categorical(IDs_train, num_classes=10)
# poses_train = poses[random_idxs]

# X_test      = X[[x for x in np.arange(100) if x not in random_idxs]]
# IDs_test  = IDs[[x for x in np.arange(100) if x not in random_idxs]]
# poses_test = IDs[[x for x in np.arange(100) if x not in random_idxs]]


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

##### TRAIN FAUST
loss_history = []

loss = nn.MSELoss() #define the loss function
opt = SGD(model.parameters() , lr = 0.03) #define the gradient descent with learning rate as 0.001

start = time.time()

for k in range(50):
  print(k)
  loss_epoch = []
  for x,y in tqdm.tqdm(zip(X, Y)):
    x = torch.unsqueeze(torch.unsqueeze(torch.tensor(x,dtype=torch.float32),0),0).cuda()
    y = torch.unsqueeze(torch.unsqueeze(torch.tensor(y,dtype=torch.float32),0),0).cuda()
    opt.zero_grad() 
    loss_value = loss(model(x), y) 
    loss_value.backward() 
    opt.step() 
    loss_epoch.append(loss_value.detach().cpu().numpy())
  loss_history.append(np.sum(loss_epoch))
end = time.time()

plt.plot(loss_history)
plt.show()

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss_value,
            }, 'chkpoint_faust_30.pth')



##### TEST FAUST

checkpoint = torch.load('chkpoint_faust_30.pth')
model.load_state_dict(checkpoint['model_state_dict'])

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
plt.show()

avg_per_class_acc = np.mean(np.diagonal(conf) / np.sum(conf, axis=1))
print('Confusion matrix:\n{}'.format(conf))

accuracy = torch.sum((Y_test_pred - Y_test)==0)/len(Y_test_pred)
print ('accuracy: ' + str(np.asarray(accuracy)))
  
  
  
############# FAUST 64 ###############

vox_faust = np.load('vox_faust_64.npy')
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(vox_faust[10])
plt.show()  
  

IDs   = np.int32(np.hstack((np.ones((10,))*0, np.ones((10,))*1, np.ones((10,))*2, np.ones((10,))*3, np.ones((10,))*4, 
                   np.ones((10,))*5, np.ones((10,))*6, np.ones((10,))*7, np.ones((10,))*8, np.ones((10,))*9)))
# IDs = to_categorical(IDs, num_classes=10)
poses = (np.repeat(np.arange(0,10)[np.newaxis],10,0)).flatten()



X, Y = shuffle(vox_faust[0:80], poses[0:80])
X_test, Y_test = shuffle(vox_faust[80:], poses[80:])



def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

Y = to_categorical(Y, num_classes=10)

  
import random 
random_idxs = np.squeeze(random.sample(range(100),80))

# X_train      = X[random_idxs]
# IDs_train   = np.int32(IDs[random_idxs])
# IDs_train = to_categorical(IDs_train, num_classes=10)
# poses_train = poses[random_idxs]

# X_test      = X[[x for x in np.arange(100) if x not in random_idxs]]
# IDs_test  = IDs[[x for x in np.arange(100) if x not in random_idxs]]
# poses_test = IDs[[x for x in np.arange(100) if x not in random_idxs]]


model = Sequential(Conv3d(1, 16, 3),   #input 64x64x64 -> 62 x 62 x 62
                   torch.nn.ReLU(),
                   Conv3d(16, 16, 5),  #input  -> 58 x 58 x 58
                   torch.nn.ReLU(),
                   Conv3d(16, 32, 10), #input  -> 49 x 49 x 49
                   torch.nn.ReLU(),
                   Conv3d(32, 64, 20), #input  -> 30 x 30 x 30
                   torch.nn.ReLU(),
                   Flatten(),
                   Linear(30 * 30 * 30 * 64,10),
                   torch.nn.Softmax()
                   ).cuda()


##### TRAIN FAUST
loss_history = []

loss = nn.MSELoss() #define the loss function
opt = SGD(model.parameters() , lr = 0.03) #define the gradient descent with learning rate as 0.001

start = time.time()

for k in range(50):
  print(k)
  loss_epoch = []
  for x,y in tqdm.tqdm(zip(X, Y)):
    x = torch.unsqueeze(torch.unsqueeze(torch.tensor(x,dtype=torch.float32),0),0).cuda()
    y = torch.unsqueeze(torch.unsqueeze(torch.tensor(y,dtype=torch.float32),0),0).cuda()
    opt.zero_grad() 
    loss_value = loss(model(x), y) 
    loss_value.backward() 
    opt.step() 
    loss_epoch.append(loss_value.detach().cpu().numpy())
  loss_history.append(np.sum(loss_epoch))
  print('Loss: ' + str(np.sum(loss_epoch)))
end = time.time()

plt.plot(loss_history)
plt.show()

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss_value,
            }, 'chkpoint_faust_64.pth')



##### TEST FAUST

checkpoint = torch.load('chkpoint_faust_64.pth')
model.load_state_dict(checkpoint['model_state_dict'])

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
plt.show()


accuracy = torch.sum((Y_test_pred - Y_test)==0)/len(Y_test_pred)
print ('accuracy: ' + str(np.asarray(accuracy)))
avg_per_class_acc = np.mean(np.diagonal(conf) / np.sum(conf, axis=1))
print('Confusion matrix:\n{}'.format(conf))

  
  
  
  
  
  
  
  
# vsize=max(o.get_max_bound()-o.get_min_bound()) * 0.005
# vsize=round(vsize,4)
        
# voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=vsize)
# bounds=voxel_grid.get_max_bound()-voxel_grid.get_min_bound()

  
# voxels=voxel_grid.get_voxels()
# vox_mesh=o3d.geometry.TriangleMesh()
# for v in voxels:
#     cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
#     cube.paint_uniform_color(v.color)
#     cube.translate(v.grid_index, relative=False)
#     vox_mesh+=cube

# o3d.visualization.draw([vox_mesh])


# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o,
#                                                             voxel_size=0.05)
# o3d.visualization.draw_geometries([voxel_grid])

# import matplotlib.pyplot as plt
# import numpy as np

# # prepare some coordinates
# x, y, z = np.indices((30, 30, 30))


# # combine the objects into a single boolean array

# # 0 : bathtub
# # 1 : Bed
# # 2 : chair
# # 3 : desk (?)
# # 4:  dresser (?)
# # 5 : monitor
# # 6 : Nighstand (?)
# # 7 : Sofas
# # 8 : Tables
# # 9 : Toilet

# idx = 28
# voxelarray = X_test[idx]
# print(Y_test[idx])
# # and plot everything
# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(voxelarray, edgecolor='k')

# plt.show()