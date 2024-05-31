# mini-batch gradient descent

import torch
import numpy as np

'''
  params
    net = neural network obj
    losf = pytorch loss obj
    optf = pytorch optimizer obj
    dataset = pytorch dataset
    training duration: batch_size & epochs
'''
def mini_batch_GD (net, losf, optf, dataset, epochs=1, batch_size=32):

  # data
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

  # training
  batches = len(trainloader)
  stat_log = batches // 9
  
  print(f'''Training configs:
    Batch-size:{batch_size}
    Training instances:{len(dataset)}
    Epochs:{epochs}
    Batches:{batches}
    Update per {stat_log} batches
  ''')

  for epoch in range(epochs):  # loop over the dataset multiple times
    print(f"------------ Epoch #{epoch+1} ------------")
    for i, data in enumerate(trainloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      # reshape so that batch_size is at the end
      inputs = inputs.view(batch_size, -1)
      # zero the parameter gradients
      optf.zero_grad()
      # forward + backward + optimize
      outputs = net(inputs)
      loss = losf(outputs, labels)
      loss.backward()
      optf.step()
      # print statistics
      if (i + 1) % stat_log == 0 or i + 1 == batches:
        print(f'Epoch: {epoch + 1} | Batches:{i + 1:5d} | Loss: {loss.item():.4f}')

  print('\nFinished Training\n')
