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
        
class FCAE(object):
    def __init__(self, input_dims, compress_dim, thres):
        self.compress_net = CN(input_dims, compress_dim).to(device)
        self.retrieve_net = RN(input_dims, compress_dim).to(device)
        self.optimiser = optim.Adam(list(self.compress_net.parameters())+list(self.retrieve_net.parameters()))
        self.criterion = nn.MSELoss()
        self.compress_dim = compress_dim
        #self.beta = beta
        self.thres = thres
        self.axes = torch.FloatTensor(get_all_axis(compress_dim, thres)).transpose(1,0).to(device)


    #Use this method to compress a dataset.
    def compress(self, data):
        #data = torch.FloatTensor(data).to(device)
        return self.compress_net(data)

    def retrieve(self, data):
        #data = torch.FloatTensor(data).to(device)
        return self.retrieve_net(data)

    #Dataset is a collection of data.
    def train(self, dataset, labels, iter, beta):
        for i in range(iter):
            loss = self.loss(dataset, labels, beta)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    #Dataset should have dimension num*channel*height*width
    def loss(self, dataset, labels, beta):
        #dataset = torch.FloatTensor(dataset).to(device)
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
            #dataset = torch.FloatTensor(dataset).to(device)
            compressed = self.compress(dataset)
            retrieved = self.retrieve(compressed)
            loss = self.criterion(dataset, retrieved)
        return loss
        
def train(model, epoches_num, beta):
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
            with torch.no_grad():
                labels = torch.FloatTensor(np.asarray(labels))/10
            labels  = Variable(labels).to(device)

            # forward + backward + optimize
            loss = model.MSE(inputs)
            with torch.no_grad():
                compressed = model.compress(inputs) #of shape [batch_size, compress_dim]
                FP_loss = model.FP(compressed, labels)
            model.train(inputs,labels, iter = 10, beta = beta)
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
                
FCAEs = []
num_models = 1
thres = 5
for i in range(num_models):
  FCAEs.append(FCAE([28,28,1], 3, thres = thres))
    
i = 0
for FCAE in FCAEs:
    i = i+1
    print(i)
    train(FCAE,1,0.02)
    
#some other training as well

bs = 1000
mnist_trainloader1 = torch.utils.data.DataLoader(mnist_trainset, batch_size=bs,
                                          shuffle=True, num_workers=2)
mnist_testloader1 = torch.utils.data.DataLoader(mnist_testset, batch_size=bs,
                                         shuffle=False, num_workers=2)
                                         
#device = torch.device('cpu')
device_c = torch.device('cpu')

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


class Fourier(object):
    def __init__(self, dim, thres, intl):
        #self.inputs = train_inputs
        #self.testdata = test_inputs
        #self.labels = labels
        self.dim = dim
        self.thres = thres  # largest absolute value of entry
        self.intl = intl
        self.thres = thres

    def get_axis(self):
        ## dim: dimension of each output vector
        ## thres:
        ## function returns all vectors of dimension dim such that each
        ## coordinate of the vector takes integer value and
        vecs, vecs1 = [], []
        ind = 0
        thres = self.thres
        dim = self.dim
        for i in range(thres * dim + 1):
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
        return torch.FloatTensor(vecs1).transpose(1,0)

    def get_mat_inv(self, train_inputs, labels):
        inputs, dim, thres = train_inputs, self.dim, 2 * self.thres + 1
        vecs = self.get_axis() ## 
        l = inputs.shape[0]
        num_row = (thres) ** dim
        m = min(l, num_row)
        instances = inputs[0:m, :]
        #print(instances.shape)
        vecs = vecs[:, 0:m]
        #print(vecs.shape)
        matrix = torch.matmul(instances, vecs)
        matrix = torch.complex(torch.FloatTensor([0]),torch.FloatTensor([1])) * matrix / self.intl
        matrix = torch.exp(matrix)
        matrix = matrix.to(device_c)
        matrix = torch.inverse(matrix)
        #matrix = matrix.to(device)

        #matrix = np.zeros([m, m], dtype=complex)
        #for i in range(m):
            #for j in range(m):
                #h = inputs[i,].reshape((dim, 1))
                #a = np.matmul(h, vecs[j])
                #exponent = complex(0, a) / self.intl
                #matrix[i, j] = cmath.exp(exponent)
        return matrix, m, vecs

    def get_Fourier_coeff(self, train_inputs, labels):
        ## k: the axis
        ## inputs, labels: the input and their labels
        
        mat, m, vecs = self.get_mat_inv(train_inputs, labels)
        l = labels[0:m].reshape([m, 1])
        l = torch.FloatTensor(np.asarray(l))
        l = torch.complex(l, torch.Tensor([0]))
        coeff = torch.matmul(mat, l)
        return coeff.reshape([m, 1]), vecs

    def get_prediction(self, train_inputs, labels, testdata):
        coeff, vecs = self.get_Fourier_coeff(train_inputs, labels)
        v = torch.matmul(testdata, vecs)
        v = torch.complex(torch.FloatTensor([0]),torch.FloatTensor([1])) * v / self.intl
        v = torch.exp(v)
        #m = coeff.shape[0]
        #n = testdata.shape[0]
        #v = np.zeros([n, m], dtype=complex)
        #for i in range(n):
            #for j in range(m):
                #t = testdata[i,].reshape((self.dim, 1))
                #a = np.matmul(t, vecs[j])
                #exponent = complex(0, a) / self.intl
                #v[i, j] = cmath.exp(exponent)
        pred = torch.matmul(v, coeff)
        return pred
        
number_epoches = 1
dim = 3
thres = 5
Four = Fourier(dim = dim, thres = thres, intl = 1)
loss = nn.CrossEntropyLoss()
num_models = 10
compress_dim = 3

#device = torch.device('cuda')
#device_c = torch.device('cpu')

def Output(x):
    x = x.reshape([-1,10])
    pred = torch.max(x, dim = 1)[1]
    return pred

def covert(labels, index):
  l = torch.FloatTensor(torch.zeros([len(labels)]))
  for i in range(len(l)):
    if labels[i] == index:
      l[i] = 1
  return l

def get_error(predicts, labels, d):
    k = 0
    a = predicts.shape[0]
    for i in range(a):
        if predicts[i] == labels[i]:
            k = k+1

    return 1 - k / d
      
def process_f(i, inputs, labels, test_inputs, test_labels):
  ## i: index of model of FCAEs
    #Outputs = []
    #errs = []
    with torch.no_grad():
        #for i in range(num_models):
        
        a = FCAEs[i].compress(inputs.to(device)).reshape([-1,compress_dim]).clone().detach().to(device_c)
        b = FCAEs[i].compress(test_inputs.to(device)).reshape([-1,compress_dim]).clone().detach().to(device_c)
    
        #Outputs.append((a,b))
            


    #for i in range(num_models):
    pred = Four.get_prediction(a, labels, b)
    pred = pred.real
    #l = loss(pred, test_labels)
    #errs.append(pred)
   
 
    return pred
    
model_ind = 0
#pred_b = []
errs = []
for j, Data in enumerate(mnist_testloader2, 0):
    # get the inputs
    test_inputs, test_labels = Data
    #test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    Preds = []
    for i, data in enumerate(mnist_trainloader1, 0):
      # get the inputs
      inputs, labels = data
      L = []
      for t in range(10):
        L.append(covert(labels, t))

      #inputs, labels = inputs.to(device), labels.to(device)
      
      for h in range(10):

        pred = process_f(model_ind, inputs = inputs, labels = L[h], test_inputs = test_inputs, test_labels = test_labels)
        if i == 0:
          #print(pred)
          Preds.append(pred.clone().detach()/(60000/bs))
        else:
          Preds[h] += pred.clone().detach()/(60000/bs)
        #preds = preds/(60000/bs)
        #Preds.append(preds.clone().detach())
    #print(Preds)
    pred_f = torch.cat(Preds, dim = 1)
    #print(pred_f)
    #print(pred_f.shape)
    pred_f = Output(pred_f)
    #print(pred_f)
    #print(test_labels)
    err = get_error(pred_f, test_labels, bs)
    print(err)
    errs.append(err)


sum(errs)/(10000/bs)

