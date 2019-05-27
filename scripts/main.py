import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from Model import Net
from ImageDataset import ImageDataset

if len(sys.argv) <= 1:
    print('Usage: python3 main.py (train|test)')

test_split = 0
shuffle_dataset = True
random_seed = 42
batch_size = 4
dataset = ImageDataset(root_dir="./dataset")
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_load = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=train_sampler)
test_load = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        sampler=test_sampler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


if sys.argv[1] == 'train':
    net.to(device)
    for epoch in range(50):
        for i, data in enumerate(train_load, 0):
            inputs, targets, name = data
            inputs = inputs.float()
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print('[%d, %5d/%d] loss: %.3f' %
                  (epoch + 1, i + 1, len(train_load), loss.item()))
    print('Finished Training')
    torch.save(net.state_dict(), './cnn')

if sys.argv[1] == 'test':
    net.load_state_dict(torch.load("./cnn", 'cpu'))
    net.eval()
    net.to(device)
    with torch.no_grad():
        for i, data in enumerate(test_load):
            print("%d/%d" % (i, len(test_load)), end="\r", flush=True)
            images, labels, name = data
            images, labels = images.float().to(device), labels.to(device)
            outputs = net(images)
            _, predicted = outputs
            c = (predicted == labels).squeeze()
            label = labels.item()
            class_correct[label] += c.item()
            class_total[label] += 1
            # print(name, label, predicted.item())
