# Autoencoder for MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


# Encoder
class Autoencoder(nn.Module):
  def __init__(self, latent_size):
    super().__init__()
    # encoder
    self.cv1 = nn.Conv2d(1, 1, 9)
    self.cv2 = nn.Conv2d(1, 1, 9)
    self.fc1 = nn.Linear(12 * 12, latent_size)
    self.ac1 = nn.Softmax()
    # decoder
    self.fc2 = nn.Linear(latent_size, 12 * 12)
    self.cv3 = nn.ConvTranspose2d(1, 1, 9)
    self.cv4 = nn.ConvTranspose2d(1, 1, 9)
    self.ac2 = nn.Softmax()
    
    
  def forward(self, x):
    # encoder
    x = self.cv1(x)
    x = self.cv2(x)
    x = x.view(1, -1)
    x = self.fc1(x)
    x = self.ac1(x)
    # latent space
    ls = x.clone().detach()
    # decoder
    x = self.fc2(x)
    x = x.view(1, 12, 12)
    x = self.cv3(x)
    x = self.cv4(x)
    x = self.ac2(x)
    return x, ls



# main function
if __name__ == '__main__':

  # data
  mnist = torchvision.datasets.MNIST('./data', download=True)
  dx = mnist.data

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
  ])

  batch_size = 100
  dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


  # neural nets
  latent_size = 2
  ae_net = Autoencoder(latent_size=latent_size)

  iter_ = iter(dataloader)

  # illustrate one sample
  # data_, targets_ = next(iter_)
  # sample = data_[0]
  # tmpo, tmpls = ae_net(sample)
  # print(tmpo.shape, tmpls.shape)

  # visualize latent space

  xs = []
  ys = []
  for features, targets in iter_:
    for x, y in zip(features, targets):
      _, lat_x = ae_net(x)
      lat_x = lat_x.view(2)
      xs.append(lat_x.tolist())
      ys.append(y.item())
    break

  xs = torch.Tensor(xs)
  ys = torch.Tensor(ys)
  x1s = xs[:, 0]
  x2s = xs[:, 1]
  labels = torch.unique(ys)
  for l in labels:
    plt.scatter(x1s[ys==l], x2s[ys==l], label=int(l.item()))
  plt.title('MNIST data in 2-dimensional latent space')
  plt.legend()
  plt.show()
 