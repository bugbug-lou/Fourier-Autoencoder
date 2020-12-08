import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import torchvision.transforms as transforms
import random
from matplotlib import pyplot as plt
import multiprocessing
from lempel_ziv_complexity import lempel_ziv_complexity
import collections
import argparse
import pickle
import os.path

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



class VCAE(object):
    def __init__(self, input_dims, compress_dim, beta):
        self.compress_net = CN(input_dims, compress_dim)
        self.retrieve_net = RN(input_dims, int(compress_dim/2))
        self.optimiser = optim.Adam(list(self.compress_net.parameters())+list(self.retrieve_net.parameters()))
        self.criterion = nn.MSELoss()
        self.compress_dim = compress_dim
        self.beta = beta

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
            if i <= int(iter/2):
                beta = 0
            else:
                beta = self.beta
            loss = self.loss(dataset, beta)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    #Dataset should have dimension num*channel*height*width
    def loss(self, dataset, beta):
        dataset = torch.FloatTensor(dataset)
        compressed = self.compress(dataset) #of shape [batch_size, compress_dim]
        k,l = compressed.shape
        mean, std = torch.split(compressed, [int(l / 2), int(l / 2)], dim=1)
        std = torch.exp(std)
        epsilon = torch.randn(mean.shape[0]).reshape([-1, 1])
        z = mean + std * epsilon
        retrieved = self.retrieve(z)
        loss = self.criterion(dataset, retrieved)
        KLdiv = self.KL(mean, std)
        if KLdiv > 0.1/self.beta:
            beta = 0
        #with torch.no_grad():
            #a = torch.abs(torch.div(KLdiv, loss))
            #b = torch.FloatTensor([self.beta])
            #beta = torch.max(b, a)
        loss = loss - KLdiv * beta
        return loss

    def KL(self, mean, std):
        epsilon = 1e-5
        KLdiv = 0.5 * (torch.square(mean) + torch.square(std) - 2.0 * torch.sum(torch.log(std + epsilon)) - 1.0)
        sign = torch.sign(torch.sum(KLdiv))
        KLdiv = sign * torch.log(sign * torch.sum(KLdiv))
        return KLdiv

    def MSE(self, dataset):
        with torch.no_grad():
            dataset = torch.FloatTensor(dataset)
            compressed = self.compress(dataset)
            k,l = compressed.shape
            mean, std = torch.split(compressed, [int(l / 2), int(l / 2)], dim=1)
            std = torch.exp(std)
            epsilon = torch.randn(mean.shape[0]).reshape([-1, 1])
            z = mean + std * epsilon
            retrieved = self.retrieve(z)
            loss = self.criterion(dataset, retrieved)
        return loss
        
VCAEs = []
MC_num = 10 #randomly initialize MC_num amount of VCAEs
compress_dim = 2
for i in range(MC_num):
    VCAEs.append(VCAE([28,28,1],compress_dim = compress_dim, beta = 0.01)) #beta = 0.01 is chosen based on grid search not shown in this code
    

def train(model, epoches_num): #train autoencoder, with epoches_num: number of loop over the training set, model: an instance of VCAE class
    for epoch in range(epoches_num):  # loop over the dataset multiple times 

        running_loss = 0.0
        running_fp = 0.0
        for i, data in enumerate(mnist_trainloader, 0):
            # get the inputs
            inputs, labels = data
            # print inputs.numpy().shape

            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels)
            inputs= Variable(inputs)

            # forward + backward + optimize
            loss = model.MSE(inputs)
            with torch.no_grad():
                compressed = model.compress(inputs) #of shape [batch_size, compress_dim]
                k,l = compressed.shape
                mean, std = torch.split(compressed, [int(l / 2), int(l / 2)], dim=1)
                std = torch.exp(std)
                KL_loss = model.KL(mean, std)
            model.train(inputs, iter = 10)
            running_loss += loss
            running_fp += KL_loss


            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print average MSE and KL loss over 100 batches
                print(str(i) + 'complete!')
                print(running_loss/100)
                print(running_fp/100)
                running_loss = 0.0
                running_fp = 0.0
          
i = 0
for model in VCAEs:
    i = i+1
    print('this is the ' + str(i) + ' model')
    train(model, 1) #train each autoencoder for 1 loop over training data, can modify to achieve better MSE loss


# The following is initialization and training of neural networks
compress_dim = 2
loss = nn.CrossEntropyLoss()
neu = 40
mean = 0
scale = 1
VCAEs1 = VCAEs #just a minor adjustment to fit the code...


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
num_models = 10

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
    #else:
        #with torch.no_grad():
            #models[i].FC1.weight = torch.nn.Parameter(models[0].FC1.weight.clone().detach())
            #models[i].FC2.weight = torch.nn.Parameter(models[0].FC2.weight.clone().detach())
            #models[i].FC3.weight = torch.nn.Parameter(models[0].FC3.weight.clone().detach())
        #optimizers.append(optim.Adam(models[i].parameters(), lr=0.1))
        
def process_t(inputs, labels):
    # the process for prediction on test data
    Outputs = []
    errs = []
    with torch.no_grad():
        for i in range(num_models):
        
            a = VCAEs1[i].compress(inputs)[:, 0:compress_dim].reshape([-1,compress_dim]).clone().detach()
        
            Outputs.append(Variable(a,requires_grad=False))
            


    for i in range(num_models):
        errs.append(get_error(models[i],Outputs[i], labels, bs))
   
 
    return errs

def process(iter, inputs, labels, models = models):
    # the process for training on training dataset
    Outputs = []
    errs = []
    with torch.no_grad():
        for i in range(num_models):
        
            a = VCAEs1[i].compress(inputs)[:, 0:compress_dim].reshape([-1,compress_dim]).clone().detach()
        
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
iter = 10  #This is generally recommended, as too large an iter will cause over training
for epoch in range(number_epoches):  # loop over the dataset multiple times 
    #errs1, errs2, errs3 = [], [], []
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
            
    a = []  #calculate the mean error the prediction method achieves on the test dataset
    for i in range(num_models):
        a.append(sum(terrs[i])/200)
    print('the mean error is: ')
    print(a)

