import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device 

## import MNIST dataset
bs = 2000 #batch_size, FCAE prefers large batch size since this facilitates more accurate Fourier coefficient calculations
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
    # tool function, get all possible arrays attainable by taking the negation of non-zero coordinates of arr (input)
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
    ## dim: dimension of each output vector (in the case of VFCAE, this corresponds to the dimension of feature vector)
    ## thres: (positive integer) the largest absolute value of entries allowed
    ## function returns all vectors of dimension dim such that each vector's entries are bounded in magnitude by thres
    ## the main purpose of this function is to produce all fourier frequency vectors bounded in magnitude by a certain threshold, in order to approximate FP norm
    vecs, vecs1 = [], []
    ind = 0
    for i in range(thres * dim +1):
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
    # this function gives the fourier coefficient corresponding to frequencies given in k of the map from x to y,
    # this is in keeping with the definition of FP norm in the original article
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
        
class VFCAE(object):
    def __init__(self, input_dims, compress_dim, thres):
        self.compress_net = CN(input_dims, 2 * compress_dim).to(device)
        self.retrieve_net = RN(input_dims, compress_dim).to(device)
        self.optimiser = optim.Adam(list(self.compress_net.parameters())+list(self.retrieve_net.parameters()))
        self.criterion = nn.MSELoss()
        self.compress_dim = compress_dim
        self.thres = thres
        self.axes = torch.FloatTensor(get_all_axis(compress_dim, thres)).transpose(1,0).to(device)


    #Use this method to compress a dataset.
    def compress(self, data):
        # compress net
        return self.compress_net(data)

    def retrieve(self, data):
        # reconstruct net
        return self.retrieve_net(data)

    #Dataset is a collection of data.
    def train(self, dataset, labels, iter, beta1, beta2):
        # beta1, beta2 are hyperparameters that weighs significance of loss terms
        for i in range(iter):
            loss = self.loss(dataset, labels, beta1, beta2)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    #Dataset should have dimension num*channel*height*width
    def loss(self, dataset, labels, beta1, beta2):
        
        compressed = self.compress(dataset) #of shape [batch_size, compress_dim]
        k,l = compressed.shape
        mean, std = torch.split(compressed, [int(l / 2), int(l / 2)], dim=1)
        KLdiv = self.KL(mean, std)
        std = torch.exp(0.5 * std)
        epsilon = torch.randn_like(std).to(device)
        z = mean + std * epsilon
        retrieved = self.retrieve(z)
        FP_loss = self.FP(mean, labels)
        loss = self.criterion(dataset, retrieved)
        loss = loss + beta1 * FP_loss + beta2 * KLdiv
        return loss

    def FP(self, compressed, labels):
        ## calculate the FP loss term 
        FP_loss = get_fc(compressed,labels,self.axes)
        return torch.sqrt(FP_loss)


    def KL(self, mean, std):
        #epsilon = 1e-10
        KLdiv = 0.5 * (torch.square(mean) + torch.exp(std) - std - 1.0).mean(dim = 0)
        return torch.sum(KLdiv)

    def MSE(self, dataset):
        with torch.no_grad():
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

## training stage
def train(model, epoches_num, beta1, beta2):
    ## model is an instance of a VFCAE autoencoder, beta1, beta2 are hyerparameters
    for epoch in range(epoches_num):  # loop over the dataset multiple times 

        running_loss = 0.0
        running_fp = 0.0
        running_kl = 0.0
        for i, data in enumerate(mnist_trainloader, 0):
            # get the inputs
            inputs, labels = data
            # print inputs.numpy().shape

            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels)
            inputs= Variable(inputs).to(device)
            with torch.no_grad():
                labels = torch.FloatTensor(np.asarray(labels))/10
            labels  = Variable(labels).to(device)

            # forward + backward + optimize
            #loss = model.MSE(inputs)
            with torch.no_grad():
                compressed = model.compress(inputs)
                k,l = compressed.shape
                mean, std = torch.split(compressed, [int(l / 2), int(l / 2)], dim=1)
                KLdiv = model.KL(mean, std)
                FP_loss = model.FP(mean, labels)
                z = mean
                retrieved = model.retrieve(z)
                loss = model.criterion(inputs, retrieved)
            model.train(inputs, labels, iter = 10, beta1 = beta1, beta2 = beta2)
            running_loss += loss
            running_fp += FP_loss
            running_kl += KLdiv

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print(str(i) + 'complete!')
                print(running_loss/10)
                print(running_fp/10)
                print(running_kl/10)
                running_loss = 0.0
                running_fp = 0.0
                running_kl = 0.0
                
#initialize VFCAEs
VFCAEs = []
num_models = 10
compress_dim = 6
thres = 3
for i in range(num_models):
    VFCAEs.append(VFCAE([28,28,1], compress_dim, thres = thres))
  
########### the above contains all information about the VFCAE part, the following contains information about neural network training part and how the two fit together####

compress_dim = 6
loss = nn.CrossEntropyLoss()
neu = 40   #number of neurons in each layer of the neural network
mean = 0    # init mean
scale = 1   #init standard dev


def Output(x):
    # function that converts output of network logits to digit recognition 0-10
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
    # the function returns an array of each NN's prediction error on test data
    Outputs = []
    errs = []
    with torch.no_grad():
        for i in range(num_models):
            a = VFCAEs[i].compress(inputs)[:, 0:compress_dim].reshape([-1,compress_dim]).clone().detach()
            Outputs.append(Variable(a,requires_grad=False))
            


    for i in range(num_models):
        errs.append(get_error(models[i],Outputs[i], labels, bs))
   
 
    return errs

def process(iter, inputs, labels, models):
    # the function trains neural networks on VFCAE feature vectors and outputs training error
    Outputs = []
    errs = []
    with torch.no_grad():
        for i in range(num_models):
            a = VFCAEs[i].compress(inputs)[:, 0:compress_dim].reshape([-1,compress_dim]).clone().detach()
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


## Now where the experiment really starts
MC_num = 5
betas1 = [0.001, 0.002, 0.002, 0.002, 0.002] #a workable hyperparameter routine
betas2 = [0.0001, 0.001, 0.002, 0.003,0.003]  #a workable hyperparameter routine
plot_datas = []  #contains the ultimate plotting datas
for t in range(MC_num):
  i = 0
  for VFCAE in VFCAEs:
      i = i+1
      print(i)
      train(VFCAE,2, betas1[t], betas2[t])
  models = [] #3  models,baseline, FP, VAE
  optimizers = []
  plot_data = []
 
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
          if i%10 == 9:
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
              if i%10 == 9:
                  print(str(i) + ' complete ')
                  #print(errs[0])
          
      a = []  #calculate the mean error the prediction method achieves on the test dataset
      for i in range(num_models):
          k = 10000/bs
          a.append(sum(terrs[i])/k)
      print('the mean error is')
      print(a)
      print(sum(a)/10)
      plot_data.append(sum(a)/num_models)

  
  del models
  del optimizers
  plot_datas.append(plot_data)

## and finally plotting:
import matplotlib.pyplot as plt
T = [1,2,3,4,5]
colors = ['darkblue','blue','lightblue','green','brown', 'purple', 'black','yellow','red','grey']
for i in range(len(plot_datas)):
  plt.plot(T, plot_datas[i], '-', color = colors[i], label = 'VFCAE ' + str(i+1) + ' sth training')
  #plt.plot(T, plot_datass[i], '--', color = colors[i])
  plt.xlabel('training epoch number')
  plt.ylabel('error rate')
plt.legend()
plt.show()
#label = 'VFCAE ' + str(i+1) + ' sth training', 
#, label ='VCAE ' + str(i+1) + ' sth training'
