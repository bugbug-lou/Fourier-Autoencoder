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


class Fourier_cs(object): #cos and sin series
    def __init__(self, dim, thres):
        #self.inputs = train_inputs
        #self.testdata = test_inputs
        #self.labels = labels
        self.dim = dim
        self.thres = thres  # largest absolute value of entry
        #self.intl = intl
        self.thres = thres
        self.axis = self.get_axis()

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
        #for v in vecs:
            #vecs1 = vecs1 + get_all(v)
        return torch.FloatTensor(vecs).transpose(1,0)

    def get_mat_inv(self, train_inputs, labels, intl):
        inputs, dim, thres = train_inputs, self.dim, self.thres + 1
        vecs = self.axis ## 
        l = inputs.shape[0]
        l = int((l+1)/2) * 2 - 1
        num_row = 2 * (thres) ** dim - 1 # sin's and cos's
        m = min(l, num_row)
        k = int(m/2) + 1
        instances = inputs[0:m, :]
        #print(instances.shape)
        vecs = vecs[:, 0:k]
        #print(vecs.shape)
        matrix = torch.matmul(instances, vecs)
        matrix = matrix * intl
        matrix_c = torch.cos(matrix)
        #print(matrix_c.shape)
        matrix_s = torch.sin(matrix)[:, 1:]
        #print(matrix_s.shape)
        Matrix = torch.cat([matrix_c, matrix_s], dim = 1)
        #print(Matrix.shape)
        Matrix = Matrix.to(device_c)
        Matrix = torch.inverse(Matrix)
        #matrix = matrix.to(device)

        #matrix = np.zeros([m, m], dtype=complex)
        #for i in range(m):
            #for j in range(m):
                #h = inputs[i,].reshape((dim, 1))
                #a = np.matmul(h, vecs[j])
                #exponent = complex(0, a) / self.intl
                #matrix[i, j] = cmath.exp(exponent)
        return Matrix, m, vecs

    def get_Fourier_coeff(self, train_inputs, labels, intl):
        ## k: the axis
        ## inputs, labels: the input and their labels
        
        mat, m, vecs = self.get_mat_inv(train_inputs, labels, intl)
        l = labels[0:m].reshape([m, 1])
        l = torch.FloatTensor(np.asarray(l))
        #l = torch.complex(l, torch.Tensor([0]))
        coeff = torch.matmul(mat, l)
        return coeff.reshape([m, 1]), vecs

    def get_prediction(self, train_inputs, labels, testdata, intl):
        coeff, vecs = self.get_Fourier_coeff(train_inputs, labels, intl)
        v = torch.matmul(testdata, vecs)
        v = v * intl
        v_c = torch.cos(v)
        v_s = torch.sin(v)[:,1:]
        vm = torch.cat([v_c, v_s], dim = 1)

        #v = torch.exp(v)

        #m = coeff.shape[0]
        #n = testdata.shape[0]
        #v = np.zeros([n, m], dtype=complex)
        #for i in range(n):
            #for j in range(m):
                #t = testdata[i,].reshape((self.dim, 1))
                #a = np.matmul(t, vecs[j])
                #exponent = complex(0, a) / self.intl
                #v[i, j] = cmath.exp(exponent)
        pred = torch.matmul(vm, coeff)
        return pred
