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
            if i <= int(iter/2):
                beta = 0
            else:
                beta = self.beta
            loss = self.loss(dataset, labels, beta)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    #Dataset should have dimension num*channel*height*width
    def loss(self, dataset, labels, beta):
        dataset = torch.FloatTensor(dataset)
        compressed = self.compress(dataset) #of shape [batch_size, compress_dim]
        FP_loss = self.FP(compressed, labels)
        retrieved = self.retrieve(compressed)
        loss = self.criterion(dataset, retrieved)
        loss = loss + beta * FP_loss
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
