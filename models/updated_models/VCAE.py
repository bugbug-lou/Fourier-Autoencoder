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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bs = 2000 #batch_size
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
    def __init__(self, input_dims, compress_dim):
        self.compress_net = CN(input_dims, compress_dim).to(device)
        self.retrieve_net = RN(input_dims, int(compress_dim/2)).to(device)
        self.optimiser = optim.Adam(list(self.compress_net.parameters())+list(self.retrieve_net.parameters()))
        self.criterion = nn.MSELoss()
        self.compress_dim = compress_dim
        #self.beta = beta

    #Use this method to compress a dataset.
    def compress(self, data):
        #data = torch.FloatTensor(data)
        return self.compress_net(data)

    def retrieve(self, data):
        #data = torch.FloatTensor(data)
        return self.retrieve_net(data)

    #Dataset is a collection of data.
    def train(self, dataset, iter, beta):
        loss = self.loss(dataset, beta)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    #Dataset should have dimension num*channel*height*width
    def loss(self, dataset, beta):
        #dataset = torch.FloatTensor(dataset)
        compressed = self.compress(dataset) #of shape [batch_size, compress_dim]
        k,l = compressed.shape
        mean, std = torch.split(compressed, [int(l / 2), int(l / 2)], dim=1)
        KLdiv = self.KL(mean, std)
        std = torch.exp(0.5 * std)
        epsilon = torch.randn_like(std).to(device)
        #epsilon = 0
        z = mean + std * epsilon
        retrieved = self.retrieve(z)
        loss = self.criterion(dataset, retrieved)
        #KL_div = 0
        #if KLdiv > 0.1/(beta + 1e-10):
            #beta = 0
        #with torch.no_grad():
            #a = torch.abs(torch.div(KLdiv, loss))
            #b = torch.FloatTensor([self.beta])
            #beta = torch.max(b, a)
        loss = loss + KLdiv * beta
        return loss

    def KL(self, mean, std):
        #epsilon = 1e-10
        KLdiv = 0.5 * (torch.square(mean) + torch.exp(std) - std - 1.0).mean(dim = 0)
        return torch.sum(KLdiv)

    def MSE(self, dataset):
        with torch.no_grad():
            #dataset = torch.FloatTensor(dataset)
            compressed = self.compress(dataset)
            k,l = compressed.shape
            mean, std = torch.split(compressed, [int(l / 2), int(l / 2)], dim=1)
            #std = torch.exp(std)
            #epsilon = torch.randn_like(std).to(device)
            #epsilon = 0
            z = mean
            retrieved = self.retrieve(z)
            loss = self.criterion(dataset, retrieved)
        return loss
        
    

def train(model, epoches_num, beta): #train autoencoder, with epoches_num: number of loop over the training set, model: an instance of VCAE class
    for epoch in range(epoches_num):  # loop over the dataset multiple times 

        running_loss = 0.0
        running_fp = 0.0
        for i, data in enumerate(mnist_trainloader, 0):
            # get the inputs
            inputs, labels = data
            # print inputs.numpy().shape

            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels)
            inputs= Variable(inputs).to(device)

            # forward + backward + optimize
            loss = model.MSE(inputs)
            with torch.no_grad():
                compressed = model.compress(inputs) #of shape [batch_size, compress_dim
                k,l = compressed.shape
                mean, std = torch.split(compressed, [int(l / 2), int(l / 2)], dim=1)
                #print(mean)
                KL_loss = model.KL(mean, std)
            model.train(inputs, iter = 10, beta = beta)
            running_loss += loss
            running_fp += KL_loss


            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print average MSE and KL loss over 100 batches
                print(str(i) + 'complete!')
                print(running_loss/10)
                print(running_fp/10)
                running_loss = 0.0
                running_fp = 0.0
        
## initialize VCAEs
VCAEs = []
num_models =  10 #randomly initialize MC_num amount of VCAEs
compress_dim_b = 12 ## to achieve an effective dimension of 6, we choose an architecture that has bottleneck layer width 12 = 2*6
for i in range(num_models):
    VCAEs.append(VCAE([28,28,1],compress_dim = compress_dim_b)) 
    
# The following is initialization and training of neural networks
compress_dim = 6 
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

        
def process_t(inputs, labels, models):
    # the process for prediction on test data
    Outputs = []
    errs = []
    with torch.no_grad():
        for i in range(num_models):
        
            a = VCAEs[i].compress(inputs)[:, 0:compress_dim].reshape([-1,compress_dim]).clone().detach()
        
            Outputs.append(Variable(a,requires_grad=False))
            


    for i in range(num_models):
        errs.append(get_error(models[i],Outputs[i], labels, bs))
   
 
    return errs

def process(iter, inputs, labels, models):
    # the process for training on training dataset
    Outputs = []
    errs = []
    with torch.no_grad():
        for i in range(num_models):
        
            a = VCAEs[i].compress(inputs)[:, 0:compress_dim].reshape([-1,compress_dim]).clone().detach()
        
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
    

MC_num = 5
betas = [0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.003, 0.003, 0.003]
plot_datas = []
for t in range(MC_num):
  i = 0
  for VCAE in VCAEs:
      i = i+1
      print(i)
      train(VCAE,2, betas[2 * t])
      train(VCAE, 2, betas[2*t + 1])
  models = [] #3  models,baseline, FP, VAE
  optimizers = []
  plot_data = []
  #errors1, errors2, errors3 = [], [], []
  #terrors1,terrors2,terrors3,terrors4 = [], [], [], []

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
      models[i] = models[i].to(device)
      #else:
          #with torch.no_grad():
              #models[i].FC1.weight = torch.nn.Parameter(models[0].FC1.weight.clone().detach())
              #models[i].FC2.weight = torch.nn.Parameter(models[0].FC2.weight.clone().detach())
              #models[i].FC3.weight = torch.nn.Parameter(models[0].FC3.weight.clone().detach())
          #optimizers.append(optim.Adam(models[i].parameters(), lr=0.1))
  number_epoches = 5
  for epoch in range(number_epoches):  # loop over the dataset multiple times
      iter = 10  #This is generally recommended, as too large an iter will cause over training
      #errs1, errs2, errs3 = [], [], []
      for i, data in enumerate(mnist_trainloader, 0):
          # get the inputs
          inputs, labels = data
          

          # wrap them in Variable
          inputs, labels = Variable(inputs, requires_grad=False), Variable(labels,requires_grad=False)
          #inputs= Variable(inputs)
          inputs, labels = inputs.to(device), labels.to(device)

          # forward + backward + optimize
          errs = process(iter, inputs, labels, models)
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
              inputs, labels = inputs.to(device), labels.to(device)

              # forward + backward + optimize
              terr = process_t(inputs, labels, models)
              for j in range(num_models):
                  terrs[j].append(terr[j])
              if i%100 == 99:
                  print(str(i) + ' complete ')
                  #print(errs[0])
          
      a = []  #calculate the mean error the prediction method achieves on the test dataset
      for i in range(num_models):
          k = 10000/bs
          a.append(sum(terrs[i])/k)
      plot_data.append(sum(a)/num_models)
      print('the mean error is')
      print(a)
      print(sum(a)/10)

  
  del models
  del optimizers
  plot_datas.append(plot_data)

import matplotlib.pyplot as plt
T = [1,2,3,4,5]
colors = ['darkblue','blue','lightblue','green','brown', 'purple', 'black','yellow','red', 'blue']
for i in range(len(plot_datas)):
  plt.plot(T, plot_datas[i], label =str(i+1) + ' sth training', color = colors[i])
  plt.xlabel('training epoch number')
  plt.ylabel('error rate')
plt.legend()
plt.show()
