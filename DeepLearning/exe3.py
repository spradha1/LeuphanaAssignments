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
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
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


# show image
def imshow(img):
  img = img / 2 + 0.5 #unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


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


  # image display
  # dataiter = iter(trainloader)
  # images, labels = next(dataiter)
  # imshow(torchvision.utils.make_grid(images))

  # # print labels
  # print(' '.join(f'{labels[j]}' for j in range(batch_size)))

  # neural network
  net = Net()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=5e-5, weight_decay=0.75)

  # training
  epochs = 5
  batches = len(trainloader)
  stat_log = batches // 9
  
  print(f'''Training configs:
    Batch-size:{batch_size}
    Training instances:{len(trainset)}
    Epochs:{epochs}
    Batches:{batches}
    Update per {stat_log} batches
  ''')

  for epoch in range(epochs):  # loop over the dataset multiple times
    print(f"------------ Epoch #{epoch+1} ------------")
    for i, data in enumerate(trainloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      # zero the parameter gradients
      optimizer.zero_grad()
      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      # print statistics
      if (i + 1) % stat_log == 0 or i + 1 == batches:
        print(f'Epoch: {epoch + 1} | Batches:{i + 1:5d} | Loss: {loss.item():4f}')

  print('\nFinished Training\n')


  # testing

  correct = 0
  total = 0

  print(f'''Testing configs
    Testing batches: {len(testloader)}
    Testing instances: {len(testset)}
  ''')
  
  with torch.no_grad(): # don't need to calculate the gradients for testing
    for data in testloader:
      images, labels = data
      # calculate outputs by running images through the network
      outputs = net(images)
      # the class with the highest energy is what we choose as prediction
      _, predicted = torch.max(outputs.data, 1)
      total += labels.numel()
      correct += (predicted == labels).sum().item()

  print(f'Accuracy on the {total} test images: {100*correct/total:3f}%')