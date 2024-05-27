# transform operations & custom dataloader

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
import torchvision.transforms as transforms


# main function
if __name__ == '__main__':

  # dataset
  cifar = CIFAR10(root='./data', train=True, download=True)
  mnist = MNIST(root='./data', train=True, download=True)
  fmist = FashionMNIST(root='./data', train=True, download=True)
