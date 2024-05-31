# dataloaders

import torch

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


# main function
if __name__ == '__main__':

  torch.manual_seed(0)

  # transforms
  random_crop = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(size=(10, 5))
  ])
  random_vertical_flip = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomVerticalFlip(p=1.0)
  ])

  # datasets
  cifar = CIFAR10(root='./data', train=True, download=True, transform=random_crop)
  mnist = MNIST(root='./data', train=True, download=True)
  fmnist = FashionMNIST(root='./data', train=True, download=True, transform=random_vertical_flip)

  # dataloaders
  cifar_loader = torch.utils.data.DataLoader(cifar, batch_size=32, num_workers=2)
  fmnist_loader = torch.utils.data.DataLoader(fmnist, batch_size=32, num_workers=2)


  # demonstrate transformations

  for i, data in enumerate(cifar_loader):
    inputs, labels = data
    print(inputs.shape)
    break
  
  dataiter = iter(fmnist_loader)
  images, labels = next(dataiter)
  plt.imshow(images[0].permute(1, 2, 0).squeeze(), cmap='gray')
  plt.axis('off')
  plt.show()
  