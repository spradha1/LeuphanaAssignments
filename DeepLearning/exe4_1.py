# tune dropout layer probability

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt



# Neural Network
class Net(nn.Module):
  def __init__(self, dp=0.5):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.drop = nn.Dropout(p=dp)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
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
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

  testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


  # neural network
  net = Net()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-5)

  dps = np.linspace(0, 1., 7)
  epochs = 15


  # tuning

  dps_dict = dict()

  print(f'''Training configs:
    Batch-size:{batch_size}
    Training instances:{len(trainset)}
    Epochs:{epochs}
  ''')

  print(f'''Testing configs:
    Testing batches: {len(testloader)}
    Testing instances: {len(testset)}
  ''')

  for p in dps:

    print(f"--------------------\nDropout layer prob: {p:.2f}")
    # training

    for epoch in range(epochs):  # loop over the dataset multiple times
      for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
      # print statistics
      print(f'Epoch: {str(epoch + 1):2} | Loss: {loss.item():.4f}') 
    

    # testing

    correct = 0
    total = 0
    
    with torch.no_grad(): # don't need to calculate the gradients for testing
      for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.numel()
        correct += (predicted == labels).sum().item()
    
    accuracy = 100*correct/total

    print(f'Test accuracy: {accuracy:.3f}%')
    dps_dict[p] = accuracy


  fig, axs = plt.subplots(1, 1, figsize=(8, 6))
  plt.plot(dps_dict.keys(), dps_dict.values())
  plt.xlabel(xlabel='Dropout layer probability')
  plt.ylabel(ylabel='Model accuracy')
  fig.suptitle('Tuning of dropout layer in Conv. NN')
  plt.tight_layout()
  plt.show()


  # inspect conv. layer weights

  layer1_weights = net.conv1.weight.data.cpu().numpy()
  print(layer1_weights, type(layer1_weights), layer1_weights.shape)
  layer2_weights = net.conv2.weight.data.cpu().numpy()
  print(layer2_weights, type(layer1_weights), layer2_weights.shape)