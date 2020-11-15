import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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


class AE(object):
    def __init__(self, input_dim, hidden_dim, compress_dim, beta):
        self.compress_net = Net(input_dim, hidden_dim, compress_dim)
        self.retrieve_net = Net(compress_dim, hidden_dim, input_dim)
        self.optimiser = optim.Adam(list(self.compress_net.parameters())+list(self.retrieve_net.parameters()))
        self.criterion = nn.MSELoss()

    #Use this method to compress a dataset.
    def compress(self, data):
        data = torch.FloatTensor(data)
        return self.compress_net(data)

    def retrieve(self, data):
        data = torch.FloatTensor(data)
        return nn.Sigmoid()(self.retrieve_net(data))

    #Dataset is a collection of data.
    def train(self, dataset, iter):
        for i in range(iter):
            loss = self.loss(dataset)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    #Compute loss
    def loss(self, dataset):
        dataset = torch.FloatTensor(dataset)
        compressed = self.compress(dataset)
        mean, std = torch.split(compressed, [len-1,1])
        std = torch.exp(std)
        epsilon = torch.random_normal(mean.shape[0])
        z = mean + std * epsilon
        retrieved = self.retrieve(z)
        loss = self.criterion(dataset, retrieved)
        KLdiv = 0.5 * (torch.square(mean) + torch.square(std) - 2.0 * torch.log(std + epsilon) - 1.0)
        return loss
