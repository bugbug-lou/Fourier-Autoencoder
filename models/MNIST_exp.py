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


def train(model, loss, optimizer, inputs, labels):
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
compress_dim = 4

for i in range(3):
    #i = i + 1
    models.append(torch.nn.Sequential())
    models[i].add_module('FC1', torch.nn.Linear(compress_dim, neu))
    models[i].add_module('relu1', torch.nn.ReLU())
    models[i].add_module('FC2', torch.nn.Linear(neu, neu))
    models[i].add_module('relu2', torch.nn.ReLU())
    models[i].add_module('FC3', torch.nn.Linear(neu, 10))
    if i == 0:
        with torch.no_grad():
            torch.nn.init.normal_(models[i].FC1.weight, mean=mean, std=scale)
            torch.nn.init.normal_(models[i].FC2.weight, mean=mean, std=scale)
            torch.nn.init.normal_(models[i].FC3.weight, mean=mean, std=scale)
        optimizers.append(optim.Adam(models[i].parameters(), lr=0.1))
    else:
        with torch.no_grad():
            models[i].FC1.weight = torch.nn.Parameter(models[1].FC1.weight.clone().detach())
            models[i].FC2.weight = torch.nn.Parameter(models[1].FC2.weight.clone().detach())
            models[i].FC3.weight = torch.nn.Parameter(models[1].FC3.weight.clone().detach())
        optimizers.append(optim.Adam(models[i].parameters(), lr=0.1))
        
def process_t(iter, inputs, labels):
    Output = []
    with torch.no_grad():
        a1 = CAE1.compress(inputs).reshape([-1,compress_dim]).clone().detach()
        a2 = VCAE1.compress(inputs)[:,0:compress_dim].reshape([-1,compress_dim]).clone().detach()
        a3 = FCAE1.compress(inputs).reshape([-1,compress_dim]).clone().detach()
        Output.append(Variable(a1))
        Output.append(Variable(a2))
        Output.append(Variable(a3))
        
    err2 = get_error(models[0],Output[0], labels, bs)
    err3 = get_error(models[1], Output[1], labels, bs)
    err4 = get_error(models[2], Output[2],labels, bs)
 
    return err2,err3, err4

def process(iter, inputs, labels, models = models):

    Output = []
    with torch.no_grad():
        a1 = CAE1.compress(inputs).reshape([-1,compress_dim]).clone().detach()
        a2 = VCAE1.compress(inputs)[:,0:compress_dim].reshape([-1,compress_dim]).clone().detach()
        a3 = FCAE1.compress(inputs).reshape([-1,compress_dim]).clone().detach()
    Output.append(a1)
    Output.append(a2)
    Output.append(a3)
    
    for j in range(iter):
        #train(models[0], loss, optimizers[0], XTrain, YTrains[num])
        #elif k == 1:
        train(models[0], loss, optimizers[0], Output[0], labels)
        #elif k == 2:
        train(models[1], loss, optimizers[1], Output[1], labels)
        #elif k == 3:
        train(models[2], loss, optimizers[2], Output[2], labels)
      
    err2 = get_error(models[0],Output[0], labels, bs)

    err3 = get_error(models[1], Output[1], labels, bs)

    err4 = get_error(models[2], Output[2],labels, bs)

    return err2,err3, err4
    
number_epoches = 1
iter = 20
for epoch in range(number_epoches):  # loop over the dataset multiple times 
    errs1, errs2, errs3 = [], [], []
    for i, data in enumerate(mnist_trainloader, 0):
        # get the inputs
        inputs, labels = data
        

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        #inputs= Variable(inputs)

        # forward + backward + optimize
        err1, err2, err3 = process(iter, inputs, labels)
        errs1.append(err1)
        errs2.append(err2)
        errs3.append(err3)
        print((err1, err2, err3))
