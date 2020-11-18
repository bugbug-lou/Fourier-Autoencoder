import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np



def get_fc(x, y, k):
    ## x: input, y: output, k:frequency
    n = x.shape[0]
    v = - 2.0 * math.pi * torch.matmul(x, k)
    v = v.reshape([1,-1])
    v1 = torch.cos(v)
    v2 = torch.sin(v)
    #v = torch.view_as_complex(v).reshape([-1,1])
    y = y.reshape([-1,1])
    coeffr = torch.matmul(v1, y) / n
    coeffi = torch.matmul(v2, y) / n
    coeff = torch.square(coeffr) + torch.square(coeffi)
    return coeff


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s):
        output = torch.FloatTensor(s)
        output = F.relu(self.layer1(output))
        output = self.layer2(output)
        return output


class FAE(object):
    def __init__(self, dataset, input_dim, hidden_dim, compress_dim,thres):
        self.compress_net = Net(input_dim, hidden_dim, compress_dim)
        self.retrieve_net = Net(compress_dim, hidden_dim, input_dim)
        self.optimiser = optim.Adam(list(self.compress_net.parameters()) + list(self.retrieve_net.parameters()))
        self.criterion = nn.MSELoss()
        #self.k = k # specifying the test function, torch.tensor
        self.thres = thres # specifying the fourier coefficients that are incorporated in the loss
        self.axes = get_all_axis(compress_dim, thres)
        #self.testfunc = torch.sin(torch.matmul(torch.FloatTensor(dataset), k))
        self.data = torch.FloatTensor(dataset)
        

    # Use this method to compress a dataset.
    def compress(self, data):
        data = torch.FloatTensor(data)
        return self.compress_net(data)

    def retrieve(self, data):
        return nn.Sigmoid()(self.retrieve_net(data))

    # Dataset is a collection of data.
    def train(self, iter, k):
        self.k = k # specifying the test function, torch.tensor
        self.testfunc = torch.sin(torch.matmul(torch.FloatTensor(self.data), k))
        k = int(iter/10)
        for i in range(iter):
            if i%k == 0:
                print('i completed')
            loss = self.loss()
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    # Compute loss
    def loss(self):
        compressed = self.compress(self.data)
        FP_loss = self.FP(compressed)
        retrieved = self.retrieve(compressed)
        loss = self.criterion(self.data, retrieved)
        loss = loss + FP_loss
        return loss

    def FP(self,compressed):
        ## thres: the axis entry maximum value
        ## k: specifying the function
        ## get_all: function to be completed
        FP_loss = torch.tensor([0.0])
        for arr in self.axes:
            arr = torch.FloatTensor(arr)
            coeff = get_fc(compressed, self.testfunc, arr)
            coeff = coeff * torch.sum(torch.square(arr))
            FP_loss = FP_loss + coeff
        return torch.sqrt(FP_loss)

    def MSE(self, dataset):
        with torch.no_grad():
            dataset = torch.FloatTensor(dataset)
            compressed = self.compress(dataset)
            #FP_loss = self.FP(dataset, compressed, self.k, self.thres)
            retrieved = self.retrieve(compressed)
            loss = self.criterion(dataset, retrieved)
        return loss



