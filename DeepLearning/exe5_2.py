# batch normalization and testing weight and input configurations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from mini_batch import mini_batch_GD


# neural network class
class FmnistNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, 64)
    self.fc2 = nn.Linear(64, 128)
    self.fc3 = nn.Linear(128, 64)
    self.fc4 = nn.Linear(64, 10)
    # initialize weights to zero, then the network doesn't learn as the gradients are zero
    # nn.init.zeros_(self.fc1.weight)
    # nn.init.zeros_(self.fc2.weight)
    # nn.init.zeros_(self.fc3.weight)
    # nn.init.zeros_(self.fc4.weight)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.fc4(x)

    # batch norm before non-linearity, better improves stability by preventing exploding & vanishing gradients by pre-normalizing
    x = F.batch_norm(x, running_mean=torch.zeros(10), running_var=torch.ones(10))

    x = F.relu(x)  # can cause exploding gradients
    # x = F.sigmoid(x) # can cause vanishing gradients

    # batch norm after non-linearity, doesn't solve the issue afterwards
    # x = F.batch_norm(x, running_mean=torch.zeros(10), running_var=torch.ones(10))

    return x
  
  # get norms of gradients of all layers
  def layer_norms(self):
    norms = []
    norms.append(torch.norm(self.fc1.weight.grad))
    norms.append(torch.norm(self.fc2.weight.grad))
    norms.append(torch.norm(self.fc3.weight.grad))
    norms.append(torch.norm(self.fc4.weight.grad))
    return norms


# main function
if __name__ == '__main__':

  trfm = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x + 100) # random constant addition
    transforms.Normalize(0, 1) # input normalization
  ])

  # data
  fmnist = FashionMNIST(root='./data', train=True, download=True, transform=trfm)
  X, y = fmnist.data, fmnist.targets
  
  # training
  fmnist_net = FmnistNN()
  
  # display norms of gradients every epoch
  # for _ in range(3):
  #   mini_batch_GD(cifar_net, nn.CrossEntropyLoss(), optim.Adam(cifar_net.parameters(), lr=1e-3, weight_decay=1e-3), dataset=fmnist)
  #   print(cifar_net.layer_norms())

  mini_batch_GD(fmnist_net, nn.CrossEntropyLoss(), optim.Adam(fmnist_net.parameters(), lr=1e-4, weight_decay=1e-4), dataset=fmnist, epochs=2)


  # testing

  batch_size = 32
  fmnist_test = FashionMNIST(root='./data', train=False, download=True, transform=trfm)
  fmnist_test_loader = torch.utils.data.DataLoader(fmnist_test, batch_size=batch_size, shuffle=True, num_workers=2)

  correct = 0
  total = 0
  print(f'''Testing configs:
    Batch-size: {batch_size}
    Testing batches: {len(fmnist_test_loader)}
    Testing instances: {len(fmnist_test)}
  ''')
  
  with torch.no_grad(): # don't need to calculate the gradients for testing
    for data in fmnist_test_loader:
      inputs, labels = data
      # reshape so that batch_size is at the end
      cur_batch_size = inputs.shape[0]
      inputs = inputs.view(cur_batch_size, -1)
      # calculate outputs by running images through the network
      outputs = fmnist_net(inputs)
      # the class with the highest energy is what we choose as prediction
      _, predicted = torch.max(outputs.data, 1)
      total += labels.numel()
      correct += (predicted == labels).sum().item()

  print(f'Accuracy on {total} test images: {100*correct/total:.2f}%')