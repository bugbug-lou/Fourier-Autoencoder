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
