#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device: ",device)
kwargs = {} if device=='cpu' else {'num_workers': 0, 'pin_memory': True}
batch_size = 10
print(kwargs)

print("\n")
print(f"CUDA avalible: {torch.cuda.is_available()}")
print(f"Current device {torch.cuda.current_device()}")
print(f"Device 0 {torch.cuda.device(0)}")
print(f"Device count {torch.cuda.device_count()}")
print(f"Device name {torch.cuda.get_device_name(0)}")
print("\n")



trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, **kwargs)

trainsize = 40000
valsize = 10000

train_set, val_set = torch.utils.data.random_split(trainset, [trainsize,valsize])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, **kwargs)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First 2D convolutional layer, taking in 3 input channel (image),
        #input size 32*32*3
        # outputting 6 convolutional features, with a square kernel size of 5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        self.dropout = nn.Dropout(0.25)
        x = self.pool(F.relu(self.conv2(x)))
        self.dropout = nn.Dropout(0.25)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(f" \n {net} \n")
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for name, param in net.named_parameters():
    print(name, '\t\t', param.shape)

print("\n")

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0): #trainloader changed to train_loader
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        correct = 0
        total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('[%d] loss: %.3f' %
            (epoch + 1, running_loss / i))
    print('Accuracy of the network on the validation set: %d %%'
          % (100 * correct / total))

print("\n")

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%'
      % (100 * correct / total))

print("\n")
#####

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

planes = []
planes_bin = []
planes_predicted = []
planes_predicted_bin = []
plane = 0

labels_list = []
predicted_list = []

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()

        labels_cpu = labels.cpu()
        predicted_cpu = predicted.cpu()

        labels_numpy = labels_cpu.numpy()
        predicted_numpy = predicted_cpu.numpy()

        for i in range(len(labels_numpy)):
            labels_list.append(labels_numpy[i])
            predicted_list.append(predicted_numpy[i])
            if (labels_numpy[i] == plane):
                planes_predicted.append(predicted_numpy[i])
                planes.append(labels_numpy[i])
                planes_bin.append(1)
                if (labels_numpy[i] == predicted_numpy[i]):
                    planes_predicted_bin.append(1)
                else:
                    planes_predicted_bin.append(0)

        #print(f"predicted_cpu: {predicted_cpu}")
        #precision, recall, thresholds = precision_recall_curve(labels_cpu, predicted_cpu)
        #print(f"labels: {labels}")
        #print(f"data: {outputs.data}")
        #print(f"predicted: {predicted}")
        #Sprint(f"c: {c}")
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
"""
print(f"planes: {planes}")
print(f"planes_predicted: {planes_predicted}")

print(f"planes_bin: {planes_bin}")
print(f"planes_predicted_bin: {planes_predicted_bin}")
"""
precision, recall, thresholds = precision_recall_curve (labels_list, predicted_list, pos_label=8)

print(f"precision: {precision}")
print(f"recall: {recall}")

"""
fig,ax=plt.subplots()
ax.step(recall,precision,color='r',alpha=0.99,where='post')
ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.show()
"""
#plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
#writer.add_figure('epoch_pr',fig,epoch)
#plt.close(fig)

#Plot PR curve
#disp = plot_precision_recall_curve(net, images, labels)
#disp.ax_.set_title('hei')

precision_list = list(0. for i in range(10))
recall_list = list(0. for i in range(10))

for i in range(10):
    precision_list[i], recall_list[i], thresholds = precision_recall_curve (labels_list, predicted_list, pos_label=i)

"""
from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
"""

plt.figure(figsize=(7, 8))
lines = []
labels = []

for i in range(10):
    l, = plt.plot(recall_list[i], precision_list[i])
    lines.append(l)
    labels.append('Precision-recall for class {0})'.format(classes[i]))

fig = plt.gcf()
#fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve for all classes')
plt.legend(lines, labels, prop=dict(size=12)) # loc=(0, -.38),

plt.show()
