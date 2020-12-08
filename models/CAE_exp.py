from matplotlib import pyplot as plt
import multiprocessing
import math
from lempel_ziv_complexity import lempel_ziv_complexity
import collections
import argparse
import pickle
import os.path
from torch.autograd import Variable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

bs = 50 #batch_size
transform = transforms.Compose(
    [transforms.ToTensor()])

mnist_trainset = datasets.MNIST(root='F:/MNISTdata', train=True,
                                        download=True, transform=transform)
mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=bs,
                                          shuffle=True, num_workers=2)
mnist_testset = datasets.MNIST(root='F:/MNISTdata', train=False,
                                       download=True, transform=transform)
mnist_testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=bs,
                                         shuffle=False, num_workers=2)



class CN(nn.Module):
    def __init__(self, input_dims, output_dim):
        #Here input_dims should be a list of length 3: height, width, channels
        super(CN, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dims[2], 16, 5, padding = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 6, 5, padding = 2)
        self.com_height = input_dims[0]//4
        self.com_width = input_dims[1]//4
        self.fc1 = nn.Linear(6*self.com_height*self.com_width, output_dim)

    #Dataset should have size num*channels*width*height
    def forward(self, x):
        l = len(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(l, 6*self.com_height*self.com_width)
        x = self.fc1(x)
        return x

class RN(nn.Module):
    def __init__(self, input_dims, output_dim):
        #Here input_dims should be a list of length 3: height, width, channels
        super(RN, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.ConvTranspose2d(6, 16, 5, stride = 2, padding = 2, output_padding = 1)
        self.conv2 = nn.ConvTranspose2d(16, input_dims[2], 5, stride = 2, padding = 2, output_padding = 1)
        self.com_height = input_dims[0]//4
        self.com_width = input_dims[1]//4
        self.fc1 = nn.Linear(output_dim, 6*self.com_height*self.com_width)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.reshape(len(x), 6, self.com_width, self.com_height)
        x = F.relu(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        return x


class CAE(object):
    def __init__(self, input_dims, compress_dim):
        self.compress_net = CN(input_dims, compress_dim)
        self.retrieve_net = RN(input_dims, compress_dim)
        self.optimiser = optim.Adam(list(self.compress_net.parameters())+list(self.retrieve_net.parameters()))
        self.criterion = nn.MSELoss()

    #Use this method to compress a dataset.
    def compress(self, data):
        data = torch.FloatTensor(data)
        return self.compress_net(data)

    def retrieve(self, data):
        data = torch.FloatTensor(data)
        return self.retrieve_net(data)

    #Dataset is a collection of data. 
    def train(self, dataset, iter):
        for i in range(iter):
            loss = self.loss(dataset)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    #Dataset should have dimension num*channel*height*width
    def loss(self, dataset):
        dataset = torch.FloatTensor(dataset)
        compressed = self.compress(dataset)
        retrieved = self.retrieve(compressed)
        loss = self.criterion(dataset, retrieved)
        return loss


#dims = [2,4,6,8,10]
CAEs = []
MC_num = 10
compress_dim = 2
for i in range(MC_num):
    CAEs.append(CAE([28,28,1],compress_dim))
    

def train(model, epoches_num):
    for epoch in range(epoches_num):  # loop over the dataset multiple times 

        running_loss = 0.0
        for i, data in enumerate(mnist_trainloader, 0):
            # get the inputs
            inputs, labels = data
            # print inputs.numpy().shape

            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels)
            inputs= Variable(inputs)

            # forward + backward + optimize
            with torch.no_grad():
                loss = model.loss(inputs)
            model.train(inputs, iter = 10)


            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(str(i) + 'complete!')
                print(running_loss/100)
                running_loss = 0.0
                
i = 0
for model in CAEs:
    i = i+1
    print('this is the ' + str(i) + ' model')
    train(model, 1)

compress_dim = 2
loss = nn.CrossEntropyLoss()
neu = 40
mean = 0
scale = 1

def Output(x):
    x = x.reshape([-1,10])
    pred = torch.max(x, dim = 1)[1]
    return pred


def Train(model, loss, optimizer, inputs, labels):
    model.train()
    inputs = Variable(inputs, requires_grad=False)
    labels = Variable(labels, requires_grad=False)
    # reset gradient
    optimizer.zero_grad()
    # forward loop
    logits = model.forward(inputs)
    output = loss.forward(logits, labels)
    # backward
    output.backward()
    optimizer.step()
    return output.item()


def get_error(model, inputs, labels, d):
    model.eval()
    inputs = Variable(inputs, requires_grad=False)
    labels = Variable(labels, requires_grad=False)
    logits = model.forward(inputs)
    predicts = Output(logits)
    a = predicts.shape[0]
    k = 0
    for i in range(a):
        if predicts[i] == labels[i]:
            k = k+1

    return 1 - k / d


def predict(model, inputs):
    model.eval()
    inputs = Variable(inputs, requires_grad=False)
    logits = model.forward(inputs)
    return logits

models = [] #3  models,baseline, FP, VAE
optimizers = []
#errors1, errors2, errors3 = [], [], []
#terrors1,terrors2,terrors3,terrors4 = [], [], [], []
loss = nn.CrossEntropyLoss()
neu = 40
mean = 0
scale = 1
compress_dim = 2
num_models = MC_num

for i in range(num_models):
    #i = i + 1
    models.append(torch.nn.Sequential())
    models[i].add_module('FC1', torch.nn.Linear(compress_dim, neu))
    models[i].add_module('relu1', torch.nn.ReLU())
    models[i].add_module('FC2', torch.nn.Linear(neu, neu))
    models[i].add_module('relu2', torch.nn.ReLU())
    models[i].add_module('FC3', torch.nn.Linear(neu, 10))
    #if i == 0:
    with torch.no_grad():
        torch.nn.init.normal_(models[i].FC1.weight, mean=mean, std=scale)
        torch.nn.init.normal_(models[i].FC2.weight, mean=mean, std=scale)
        torch.nn.init.normal_(models[i].FC3.weight, mean=mean, std=scale)
    optimizers.append(optim.Adam(models[i].parameters(), lr=0.1))
   # else:
        #with torch.no_grad():
            #models[i].FC1.weight = torch.nn.Parameter(models[1].FC1.weight.clone().detach())
            #models[i].FC2.weight = torch.nn.Parameter(models[1].FC2.weight.clone().detach())
            #models[i].FC3.weight = torch.nn.Parameter(models[1].FC3.weight.clone().detach())
        #optimizers.append(optim.Adam(models[i].parameters(), lr=0.1))
        
def process_t(inputs, labels):
    Outputs = []
    errs = []
    with torch.no_grad():
        for i in range(num_models):
        
            a = CAEs[i].compress(inputs).reshape([-1,compress_dim]).clone().detach()
        
            Outputs.append(Variable(a,requires_grad=False))
            


    for i in range(num_models):
        errs.append(get_error(models[i],Outputs[i], labels, bs))
   
 
    return errs

def process(iter, inputs, labels, models = models):

    Outputs = []
    errs = []
    with torch.no_grad():
        for i in range(num_models):
        
            a = CAEs[i].compress(inputs).reshape([-1,compress_dim]).clone().detach()
        
            Outputs.append(Variable(a,requires_grad=False))
    
    for i in range(num_models):
        errs.append(get_error(models[i],Outputs[i], labels, bs))
    
    for j in range(iter):
        #train(models[0], loss, optimizers[0], XTrain, YTrains[num])
        #elif k == 1:
        for i in range(num_models):
            Train(models[i], loss, optimizers[i], Outputs[i], labels)
            err = get_error(models[i],Outputs[i], labels, bs)
            if err == 0:
                break
        
    
    return errs
    
number_epoches = 1
iter = 10
for epoch in range(number_epoches):  # loop over the dataset multiple times 
    errs1, errs2, errs3 = [], [], []
    for i, data in enumerate(mnist_trainloader, 0):
        # get the inputs
        inputs, labels = data
        

        # wrap them in Variable
        inputs, labels = Variable(inputs, requires_grad=False), Variable(labels,requires_grad=False)
        #inputs= Variable(inputs)

        # forward + backward + optimize
        errs = process(iter, inputs, labels)
        print(errs)
        if i%100 == 99:
            print(str(i) + ' complete ')
    
    number_epoches = 1
    #iter = 50
    for epoch in range(number_epoches):  # loop over the dataset multiple times 
        terrs = []
        for i in range(num_models):
            terrs.append([])
        for i, data in enumerate(mnist_testloader, 0):
            # get the inputs
            inputs, labels = data


            # wrap them in Variable
            inputs, labels = Variable(inputs, requires_grad=False), Variable(labels,requires_grad=False)
            #inputs= Variable(inputs)

            # forward + backward + optimize
            terr = process_t(inputs, labels)
            for j in range(num_models):
                terrs[j].append(terr[j])
            if i%100 == 99:
                print(str(i) + ' complete ')
                #print(errs[0])
            
    a = []
    for i in range(num_models):
        a.append(sum(terrs[i])/200)
    print('the mean error is')
    print(a)


