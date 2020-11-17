import tensorflow as tf
from tensorflow import keras
from tensorflow import math
from keras.optimizers import Adam
from keras.layers import Input, LeakyReLU, Conv2D, ReLU, BatchNormalization, UpSampling2D, \
    MaxPooling2D, AveragePooling2D, Flatten, Dense, LSTM, Concatenate
from keras.activations import softmax
from keras.models import Model
'''
class AE(object):
    def __init__(self):

    def compress(self):
        ### the first half of the model: compress data to bottle-neck
    
    def retrieve(self):
        ### the second half of the model: from bottleneck regain data

    def train(self):

    def loss(self):

    def get_rep(self, inputs):
        ### attain bottle-neck representation of input data: inputs
'''
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

    def forward(self, s):
        output = torch.FloatTensor(s)
        output = F.relu(self.layer1(output))
        output = self.layer2(output)
        return output

class AE(object):
    def __init__(self, input_dim, hidden_dim, compress_dim):
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
        return self.retrieve_net(data)

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
        retrieved = self.retrieve(compressed)
        loss = self.criterion(dataset, retrieved)
        return loss

