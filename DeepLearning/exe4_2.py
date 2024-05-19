# Logistic regression in pytorch for MNIST 1s & 7s

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset

import torchvision
import torchvision.transforms as transforms



# Neural Network
class Lognet(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 4, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(4, 8, 5)
    self.fc1 = nn.Linear(8 * 4 * 4, 64)
    self.fc2 = nn.Linear(64, 1)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.sigmoid(self.fc2(x))
    return x



# main function
if __name__ == '__main__':

  # prepare data

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
  ])
  batch_size = 32
  trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  
  # filter 1s and 7s

  tr_y = trainset.targets
  tr_ids1n7 = [i for i, sample in enumerate(tr_y) if sample in [1, 7]]
  trainset.targets[tr_y == 1] = 0
  trainset.targets[tr_y == 7] = 1
  trainset = Subset(trainset, tr_ids1n7)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

  ts_y = testset.targets
  ts_ids1n7 = [i for i, sample in enumerate(ts_y) if sample in [1, 7]]
  testset.targets[ts_y == 1] = 0
  testset.targets[ts_y == 7] = 1
  testset = Subset(testset, ts_ids1n7)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  # model
  lognet = Lognet()
  learning_rate = 5e-4
  criterion = nn.BCELoss()
  optimizer = torch.optim.Adam(lognet.parameters(), lr=learning_rate)
  epochs = 15


  # training

  print(f'''Training configs:
    Batch-size:{batch_size}
    Training instances:{len(trainset)}
    Epochs:{epochs}
  ''')
  for epoch in range(epochs):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      labels = labels.to(torch.float32)
      # zero the parameter gradients
      optimizer.zero_grad()
      # forward + backward + optimize
      outputs = lognet(inputs)
      loss = criterion(outputs.reshape(-1, ), labels)
      loss.backward()
      optimizer.step()
    # print statistics
    print(f'Epoch: {str(epoch + 1):2} | Loss: {loss.item():.6f}') 
    
  # testing
  print(f'''\nTesting configs:
    Testing batches: {len(testloader)}
    Testing instances: {len(testset)}
  ''')
  correct = 0
  total = 0
  
  with torch.no_grad(): # don't need to calculate the gradients for testing
    for data in testloader:
      images, labels = data
      # calculate outputs by running images through the network
      outputs = lognet(images)
      # the class with the highest energy is what we choose as prediction
      predicted = outputs.data.flatten().apply_(lambda x: 1 if x > 0.5 else 0)
      total += labels.numel()
      correct += (predicted == labels).sum().item()
  
  accuracy = 100*correct/total
  print(f'Test accuracy: {accuracy:.3f}%')

  # save model
  PATH = './models/exe4_2_lognet.pt'
  torch.save(lognet.state_dict(), PATH)