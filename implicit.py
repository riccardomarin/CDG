import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import List

import imageio
from skimage.transform import resize
from skimage.color import rgba2rgb
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

!pip install livelossplot --quiet
from livelossplot import PlotLosses

image = imageio.imread('https://play-lh.googleusercontent.com/pUxz7N3Jb9vLzKNAWPR5psitbwu08q5kPDiJeI3f8CeeNXEHi0qgD6C3aFW83aAPDA')[...]
image = rgba2rgb(resize(image, (512, 512)))
plt.imshow(image);
plt.show()

image = torch.tensor(image, dtype=torch.float32)

x = np.linspace(0, 1, image.shape[1], endpoint=False)

# sample the grid, which will be the input to the model
grid = torch.tensor(np.stack(np.meshgrid(x, x), -1), dtype=torch.float32)
X, Y = [grid.view(-1, 2), image.view(-1, 3)]
test_X, test_y = [X[1::2], Y[1::2]] 
train_X, train_y = [X[::2], Y[::2]]

test_X.requires_grad = False
train_X.requires_grad = False

class NeuralField(nn.Module):
  def __init__(self, hidden_layers=2, neurons_per_layer=256, input_dimension=2):
    super().__init__()
    self.input_layer = nn.Linear(input_dimension, neurons_per_layer)
    self.hidden_layers = nn.ModuleList([nn.Linear(neurons_per_layer, neurons_per_layer) for i in range(hidden_layers)])
    self.output_layer = nn.Linear(neurons_per_layer, 3)

  def forward(self, input):
    x = F.relu(self.input_layer(input))
    for layer in self.hidden_layers:
      x = F.relu(layer(x))
    return torch.sigmoid(self.output_layer(x))

def mse(gt, pred):
  return 0.5 * torch.mean((gt - pred) ** 2., (-1, -2)).sum(-1).mean()

def psnr(gt, pred):
  return -10 * torch.log10(2. * torch.mean((gt - pred) ** 2.))

model = nn.DataParallel(NeuralField().cuda())
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
images = []

liveloss = PlotLosses()

for i in range(2000):
  print(i)
  model.train()
  optimizer.zero_grad(set_to_none=True)
  prediction = model(train_X)
  loss = mse(train_y.to('cuda'), prediction)
  loss.backward()
  optimizer.step()

  if i % 25 == 0:
    with torch.no_grad():
      model.eval()
      reconstruction = model(X).detach().cpu()
    
    # liveloss.update({'PSNR train': psnr(train_y, prediction.detach().cpu()), 
    #                  'Loss train': mse(train_y, prediction.detach().cpu()),
    #                  'PSNR test': psnr(test_y, reconstruction[::2]),
    #                  'Loss test': mse(test_y, reconstruction[::2])}, 
    #                 current_step=i)
    # liveloss.send()
    images.append(reconstruction.numpy().reshape(512, 512, 3))
    
model.eval()
pred = model(X.to('cuda')).cpu().detach().numpy().reshape(512, 512, 3)
f, (ax0, ax1) = plt.subplots(1, 2)

ax0.imshow(pred)
ax1.imshow(image);
plt.show()

###########

im_dim = 128
image = imageio.imread('https://play-lh.googleusercontent.com/pUxz7N3Jb9vLzKNAWPR5psitbwu08q5kPDiJeI3f8CeeNXEHi0qgD6C3aFW83aAPDA')[...]
#image = imageio.imread('https://icon-icons.com/downloadimage.php?id=142947&root=2348/PNG/512/&file=x_warning_badged_outline_icon_142947.png')[...]
image = rgba2rgb(resize(image, (im_dim, im_dim)))
# plt.imshow(image);
# plt.show()

image = torch.tensor(image, dtype=torch.float32)

x = np.linspace(0, 1, image.shape[1], endpoint=False)

# sample the grid, which will be the input to the model
grid = torch.tensor(np.stack(np.meshgrid(x, x), -1), dtype=torch.float32)
X, Y = [grid.view(-1, 2), image.view(-1, 3)]
test_X, test_y = [X[1::2], Y[1::2]] 
train_X, train_y = [X[::2], Y[::2]]

test_X.requires_grad = False
train_X.requires_grad = False

FOURIER_DIM = 512
FOURIER_SCALE = 1.
INPUT_DIMS = 2 * FOURIER_DIM

B = FOURIER_SCALE * torch.randn(size=(2, FOURIER_DIM), requires_grad=False)

def apply_fourier_features(x, B):
  projection = (2 * np.pi * x) @ B 
  transformed = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
  return transformed

model = nn.DataParallel(NeuralField(input_dimension=INPUT_DIMS).cuda())
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
ff_images = []

for i in range(3000):
  print(i)
  optimizer.zero_grad(set_to_none=True)
  prediction = model(apply_fourier_features(train_X,  B))
  loss = mse(train_y.to('cuda'), prediction)
  loss.backward()
  optimizer.step()

  if i % 25 == 0:
    with torch.no_grad():
      optimizer.zero_grad(set_to_none=True)
      reconstruction = model(apply_fourier_features(X, B)).detach().cpu()

    ff_images.append(reconstruction.cpu().detach().numpy().reshape(im_dim, im_dim, 3))
   
    
    
model.eval()

predicted_image = model(apply_fourier_features(X, B).to('cuda')).cpu().detach().numpy()

predicted_image = predicted_image.reshape(image.shape)

factor=4
new_shape = (image.shape[0]*factor, image.shape[1]*factor,3)
x = np.linspace(0, 1, image.shape[1]*factor, endpoint=False)

# sample the grid, which will be the input to the model
grid = torch.tensor(np.stack(np.meshgrid(x, x), -1), dtype=torch.float32)
X = grid.view(-1, 2)
super_res_predicted_image = model(apply_fourier_features(X, B).to('cuda')).cpu().detach().numpy()
super_res_predicted_image = super_res_predicted_image.reshape(new_shape)

plt.subplot(121)
plt.imshow(predicted_image);

plt.subplot(122)
plt.imshow(super_res_predicted_image);
plt.show()



##########


vox_faust = np.load('vox_faust_30.npy')
image = torch.tensor(vox_faust[6], dtype=torch.float32).unsqueeze(-1)
images = np.stack((image,image,image),-1)

class NeuralField(nn.Module):
  def __init__(self, hidden_layers=4, neurons_per_layer=512, input_dimension=2):
    super().__init__()
    self.input_layer = nn.Linear(input_dimension, neurons_per_layer)
    self.hidden_layers = nn.ModuleList([nn.Linear(neurons_per_layer, neurons_per_layer) for i in range(hidden_layers)])
    self.output_layer = nn.Linear(neurons_per_layer, 3)

  def forward(self, input):
    x = F.relu(self.input_layer(input))
    for layer in self.hidden_layers:
      x = F.relu(layer(x))
    return torch.sigmoid(self.output_layer(x))


x = np.linspace(0, 1, image.shape[1], endpoint=False)

# sample the grid, which will be the input to the model
grid = torch.tensor(np.stack(np.meshgrid(x, x, x), -1), dtype=torch.float32)


X, Y = [grid.view(-1, 3), image.view(-1, 1)]
test_X.requires_grad = False
train_X.requires_grad = False

FOURIER_DIM = 512
FOURIER_SCALE = 1.
INPUT_DIMS = 2 * FOURIER_DIM

B = FOURIER_SCALE * torch.randn(size=(3, FOURIER_DIM), requires_grad=False)

def apply_fourier_features(x, B):
  projection = (2 * np.pi * x) @ B 
  transformed = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
  return transformed

model = nn.DataParallel(NeuralField(input_dimension=INPUT_DIMS).cuda())
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
ff_images = []

for i in range(1500):
  print(i)
  optimizer.zero_grad(set_to_none=True)
  prediction = model(apply_fourier_features(train_X,  B))
  loss = mse(train_y.to('cuda'), prediction)
  loss.backward()
  optimizer.step()

  if i % 25 == 0:
    with torch.no_grad():
      optimizer.zero_grad(set_to_none=True)
      reconstruction = model(apply_fourier_features(X, B)).detach().cpu()

    
    
model.eval()
predicted_image = model(apply_fourier_features(X, B).to('cuda')).cpu().detach().numpy()

predicted_image = predicted_image.reshape(images.shape)
# plt.imshow(predicted_image);
# plt.show()

##########

voxelarray = np.squeeze(predicted_image[:,:,:,:,0])
vox = voxelarray.reshape((predicted_image.shape[0],predicted_image.shape[0],predicted_image.shape[0]))
vox[vox<0.5] = 0


# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray, edgecolor='k')

plt.show()


new_shape = (image.shape[0]*2, image.shape[1]*2, image.shape[2]*2,3)
x = np.linspace(0, 1, image.shape[1]*2, endpoint=False)

# sample the grid, which will be the input to the model
grid = torch.tensor(np.stack(np.meshgrid(x, x, x), -1), dtype=torch.float32)
X = grid.view(-1, 3)
predicted_image = model(apply_fourier_features(X, B).to('cuda')).cpu().detach().numpy()
predicted_image = predicted_image.reshape(new_shape)


voxelarray = np.squeeze(predicted_image[:,:,:,0])

vox = voxelarray.reshape((predicted_image.shape[0],predicted_image.shape[0],predicted_image.shape[0]))
vox[vox<0.5] = 0

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray, edgecolor='k')
plt.show()
