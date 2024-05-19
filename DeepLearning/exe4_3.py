# Add L1 & L2 regularizers to and tune the model from exe4_2 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from exe4_2 import Lognet


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
  # weight decay parameter enacts L2 regularization in ADAM optimizer
  optimizer = torch.optim.Adam(lognet.parameters(), lr=learning_rate, weight_decay=1e-2)
  epochs = 25


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
      # l1 regularization on the second fully connected layer
      params = torch.cat([x.view(-1) for x in lognet.fc2.parameters()])
      loss += torch.linalg.norm(params, ord=1)
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
  PATH = './models/exe4_3_lognet.pt'
  torch.save(lognet.state_dict(), PATH)