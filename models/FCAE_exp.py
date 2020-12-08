import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

bs = 1000 #batch_size
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

def get_all(arr):
    # get all possible arrays given by taking tha negation of some coordinates of arr
    dim = arr.shape[0]
    vec = []
    for i in range(dim):
        if i == 0:
            if arr[i] != 0:
                k = int(arr[0])
                vec.append(np.array([k]))
                vec.append(np.array([-k]))
            else:
                vec.append(np.array([0]))

        else:
            if int(arr[i]) != 0:
                for j in range(len(vec)):
                    v = vec[j]
                    l = np.copy(v)
                    k = int(arr[i])
                    k = np.array([k])
                    vec[j] = np.concatenate((v, k))
                    l = np.concatenate((l, -k))
                    vec.append(l)
            else:
                for i in range(len(vec)):
                    v = vec[i]
                    vec[i] = np.concatenate((v, np.array([0])))
    return vec

def get_all_axis(dim, thres):
    ## dim: dimension of each output vector
    ## thres:
    ## function returns all vectors of dimension dim such that each
    ## coordinate of the vector takes integer value and
    vecs, vecs1 = [], []
    ind = 0
    for i in range(thres * dim):
        if i == 0:
            vecs.append(np.zeros(dim))
        else:
            k = len(vecs)
            c = set([])
            for h in range(ind, k):
                l = vecs[h]
                for j in range(dim):
                    if l[j] < thres:
                        f = np.copy(l)
                        f[j] = f[j] + 1
                        f = f.tostring()
                        c.add(f)
            ind = k
            for element in c:
                element = np.fromstring(element)
                vecs.append(element)
    for v in vecs:
        vecs1 = vecs1 + get_all(v)
    return vecs1

def get_fc(x, y, k):
    ## x: input, y: output, k:frequency matrix (data dim first), w: weight given by the frequency axes
    bs,cd = x.shape
    k = k.reshape([cd,-1]) #cd*num of frequencies
    w = torch.sum(torch.square(k), dim = 0) #length num
    v = - 2.0 * math.pi * torch.matmul(x, k) #bs*num of freq
    #v = v.reshape([1,-1])
    v1 = torch.cos(v)
    v2 = torch.sin(v)
    #v = torch.view_as_complex(v).reshape([-1,1])
    y = y.reshape([1,-1])
    coeffr = torch.matmul(y, v1) / bs  #i*num of freq
    coeffi = torch.matmul(y, v2) / bs
    coeff = torch.square(coeffr) + torch.square(coeffi)
    w = w.reshape([-1,1]) #numof freq*1
    coeff = torch.matmul(coeff, w)
    return coeff

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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


class FCAE(object):
    def __init__(self, input_dims, compress_dim, beta, thres):
        self.compress_net = CN(input_dims, compress_dim)
        self.retrieve_net = RN(input_dims, compress_dim)
        self.optimiser = optim.Adam(list(self.compress_net.parameters())+list(self.retrieve_net.parameters()))
        self.criterion = nn.MSELoss()
        self.compress_dim = compress_dim
        self.beta = beta
        self.thres = thres
        self.axes = torch.FloatTensor(get_all_axis(compress_dim, thres)).transpose(1,0)


    #Use this method to compress a dataset.
    def compress(self, data):
        data = torch.FloatTensor(data)
        return self.compress_net(data)

    def retrieve(self, data):
        data = torch.FloatTensor(data)
        return self.retrieve_net(data)

    #Dataset is a collection of data.
    def train(self, dataset, labels, iter):
        for i in range(iter):
            loss = self.loss(dataset, labels)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    #Dataset should have dimension num*channel*height*width
    def loss(self, dataset, labels):
        dataset = torch.FloatTensor(dataset)
        compressed = self.compress(dataset) #of shape [batch_size, compress_dim]
        FP_loss = self.FP(compressed, labels)
        retrieved = self.retrieve(compressed)
        loss = self.criterion(dataset, retrieved)
        loss = loss + self.beta * FP_loss
        return loss

    def FP(self, compressed, labels):
        ## thres: the axis entry maximum value
        ## k: specifying the function
        ## get_all: function to be completed
        FP_loss = get_fc(compressed,labels,self.axes)
        return torch.sqrt(FP_loss)


    def MSE(self, dataset):
        with torch.no_grad():
            dataset = torch.FloatTensor(dataset)
            compressed = self.compress(dataset)
            retrieved = self.retrieve(compressed)
            loss = self.criterion(dataset, retrieved)
        return loss

from torch.autograd import Variable
import math
def train(model, epoches_num):
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
            with torch.no_grad():
                labels = torch.FloatTensor(np.asarray(labels))/10
            labels  =Variable(labels)

            # forward + backward + optimize
            loss = model.MSE(inputs)
            with torch.no_grad():
                compressed = model.compress(inputs) #of shape [batch_size, compress_dim]
                FP_loss = model.FP(compressed, labels)
            model.train(inputs,labels, iter = 10)
            running_loss += loss
            running_fp += FP_loss

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print(str(i) + 'complete!')
                print(running_loss/10)
                print(running_fp/10)
                running_loss = 0.0
                running_fp = 0.0
                
#betas = [0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2]
FCAEs = []
MC_num = 10 #can adjust this to be bigger, but take a loooot of training time
for i in range(MC_num):
    FCAEs.append(FCAE([28,28,1],compress_dim = 2, beta = 0.09, thres = 10))
    
i = 0
for FCAE in FCAEs:
    i = i+1
    print(i)
    train(FCAE,1)
    
compress_dim = 2
loss = nn.CrossEntropyLoss()
neu = 40
mean = 0
scale = 1

import numpy as np
import datetime
import random
from matplotlib import pyplot as plt
import multiprocessing
import torch
from torch import optim
from torch.autograd import Variable
import math
from lempel_ziv_complexity import lempel_ziv_complexity
import collections
import argparse
import pickle
import os.path

def array_to_string(x):
    y = ''
    for l in x:
        y += str(int(l))
    return y


def Output(x):
    x = x.reshape([-1,10])
    pred = torch.max(x, dim = 1)[1]
    return pred


def get_max_freq(x):
    T = collections.Counter(x)
    Y = np.array(list(T.values()), dtype=np.longfloat)
    a = np.max(Y)
    for f in list(T.keys()):
        if T[f] == a:
            return f


def get_LVComplexity(x):
    ones = torch.ones(len(x))
    zeros = torch.zeros(len(x))
    if torch.all(torch.eq(x, ones)) or torch.all(torch.eq(x, ones)):
        return np.log2(len(x))
    else:
        with torch.no_grad():
            a = N_w(x)
            y = np.asarray(x)
            y = y[::-1]
            b = N_w(y)
        return np.log2(len(x)) * (a + b) / 2


def N_w(S):
    # get number of words in dictionary
    i = 0
    C = 1
    u = 1
    v = 1
    vmax = v
    while u + v < len(S):
        if S[i + v] == S[u + v]:
            v = v + 1
        else:
            vmax = max(v, vmax)
            i = i + 1
            if i == u:
                C = C + 1
                u = u + vmax
                v = 1
                i = 0
                vamx = v
            else:
                v = 1
    if v != 1:
        C = C + 1
    return C


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
num_models = len(FCAEs)

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
    Outputs = []
    errs = []
    with torch.no_grad():
        for i in range(num_models):
        
            a = FCAEs[i].compress(inputs).reshape([-1,compress_dim]).clone().detach()
        
            Outputs.append(Variable(a,requires_grad=False))
            


    for i in range(num_models):
        errs.append(get_error(models[i],Outputs[i], labels, bs))
   
 
    return errs

def process(iter, inputs, labels, models = models):

    Outputs = []
    errs = []
    with torch.no_grad():
        for i in range(num_models):
        
            a = FCAEs[i].compress(inputs).reshape([-1,compress_dim]).clone().detach()
        
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
            
    a = []
    for i in range(num_models):
        a.append(sum(terrs[i])/10)
    print('the mean error is')
    print(a)


