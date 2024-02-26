import os
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#landclass = {'Farmland':0,'Forest':1,'Industrial':2,'Residential':3,'River':4}
#landclass = {'farmland':0,'forest':1,'industrial':2,'denseresidential':3,'river':4}
#landclass = {'farmland':0,'forest':1,'industrial':2,'residential':3,'river':4}
landclass = {'Farmland':0,'Woods':1,'Industrial':2,'Residential':3,'Rural':4,'RiverLake':5}
#landclass = {'Farmland':0,'Rural':1}
landkeys = list(landclass.keys())

trainPath = 'Data/ChangSha/newtrain/newAll'
testPath = 'Data/ChangSha/newtest/newAll'
errorPath = 'Data/ChangSha/errors/error_newChangSha/'
modelPath = 'Models/effnet_newChangSha.pth'

learingRate = 0.001
lambdaRate = 0.002
momentumRate = 0


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
     
class EffNet(nn.Module):

    def __init__(self, nb_classes=6, include_top=True, weights=None):
        super(EffNet, self).__init__()
        
        self.block1 = self.make_layers(32, 64)
        self.block2 = self.make_layers(64, 128)
        self.block3 = self.make_layers(128, 256)
        self.flatten = Flatten()
        self.linear = nn.Linear(16384, nb_classes)
        self.include_top = include_top
        self.weights = weights

    def make_layers(self, ch_in, ch_out):
        layers = [
            nn.Conv2d(3, ch_in, kernel_size=(1,1), stride=(1,1), bias=False, padding=0, dilation=(1,1)) if ch_in ==32 else nn.Conv2d(ch_in, ch_in, kernel_size=(1,1),stride=(1,1), bias=False, padding=0, dilation=(1,1)) ,
            self.make_post(ch_in),
            # DepthWiseConvolution2D
            nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(1, 3),stride=(1,1), padding=(0,1), bias=False, dilation=(1,1)),
            self.make_post(ch_in),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            # DepthWiseConvolution2D
            nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(3, 1), stride=(1,1), padding=(1,0), bias=False, dilation=(1,1)),
            self.make_post(ch_in),
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 2), stride=(1, 2), bias=False, padding=(0,0), dilation=(1,1)),
            self.make_post(ch_out),
        ]
        return nn.Sequential(*layers)

    def make_post(self, ch_in):
        layers = [
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(ch_in, momentum=0.99)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.linear(x)
        return x

class MyDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_index = self.images[idx]
        image_path = os.path.join(self.img_dir, image_index)
        image = read_image(image_path)
        land = image_path.split('\\')[-1].split('.')[0].split('_')[0]
        label = landclass[land]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainData = MyDataset(trainPath, transform)
trainloader = DataLoader(trainData, batch_size=64, shuffle=True)

testData = MyDataset(testPath, transform)
testloader = DataLoader(testData, batch_size=64, shuffle=True)

net = EffNet().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learingRate,weight_decay=lambdaRate)
batch_size = 64

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

pictureindex = 0

def SaveResult(inputs,outputs,labels):
    global pictureindex
    keys = list(landclass.keys())
    _, classindex = torch.max(outputs.data, 1)

    for i in range(0,len(inputs)):
        index = classindex[i]
        classlabel = labels[i]
        if index != classlabel:
            image = inputs[i] / 2 + 0.5
            classresult = landkeys[index]
            image_path = errorPath + landkeys[classlabel] + '_' + classresult+'_{}.jpg'.format(pictureindex)
            pictureindex += 1
            save_image(image,image_path)

def countNumber(errorPath):
    pass
        

epochs = 1


correct_pred = {classname: 0 for classname in landclass}
total_pred = {classname: 0 for classname in landclass}
acc_writer = SummaryWriter('./logs')
def calcAcc(epoch):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            imgs = data[0]
            # calculate outputs by running images through the network
            outputs = net(images)

            if epoch == (epochs-1):
                SaveResult(imgs,outputs,labels)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[landkeys[label]] += 1
                total_pred[landkeys[label]] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(correct, total)
    acc_writer.add_scalar('Test/Acc',100 * correct / total,epoch)
    print(f'Accuracy of the network on the {len(testData.images)} test images: {100 * correct / total:.2f} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

loss_writer = SummaryWriter('./logs')
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
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
        # if i % 40 == 39:  # print every 2 mini-batches
        #     print(f'[{epoch + 1}, {(i + 1)*batch_size:5d}] loss: {running_loss / 40:.3f}')
        #     running_loss = 0.0
    loss_writer.add_scalar('Test/Loss',running_loss,epoch)
    calcAcc(epoch)

print('Finished Training')

torch.save(net,modelPath)

# dataiter = iter(testloader)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
