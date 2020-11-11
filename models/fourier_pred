import cmath
from scipy.fft import fft, ifft
import numpy as np
import datetime
import random
from matplotlib import pyplot as plt
import multiprocessing
import torch
from torch.autograd import Variable
from torch import optim
from lempel_ziv_complexity import lempel_ziv_complexity
import collections
import argparse

class Fourier(object):
    def __init__(self, dim, thres, train_inputs, test_inputs, labels):
        self.inputs = train_inputs
        self.testdata = test_inputs
        self.labels = labels
        self.dim = dim
        self.thres = thres

    def get_axis(self):
        ## dim: dimension of each output vector
        ## thres:
        ## function returns all vectors of dimension dim such that each
        ## coordinate of the vector takes integer value and
        vecs = []
        ind = 0
        thres = self.thres
        dim = self.dim
        for i in range(thres * dim):
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
        return vecs

    def get_mat_inv(self):
        inputs, dim, thres = self.inputs, self.dim, self.thres
        vecs = self.get_axis()
        l = inputs.shape[0]
        num_row = (thres) ** dim
        m = min(l, num_row)
        matrix = np.zeros([m, m], dtype=complex)
        for i in range(m):
            for j in range(m):
                h = inputs[i, ].reshape((dim, 1))
                a = np.matmul(h, vecs[j])
                exponent = complex(0, a)
                matrix[i, j] = cmath.exp(exponent)
        return np.linalg.pinv(matrix), m, vecs

    def get_Fourier_coeff(self):
        ## k: the axis
        ## inputs, labels: the input and their labels
        labels = self.labels
        mat, m, vecs = self.get_mat_inv()
        l = labels[0:m].reshape([m, 1])
        coeff = np.matmul(mat, l)
        return coeff.reshape([m, 1]), vecs

    def get_prediction(self):
        coeff, vecs = self.get_Fourier_coeff()
        testdata = self.testdata
        m = coeff.shape[0]
        n = testdata.shape[0]
        v = np.zeros([n, m], dtype=complex)
        for i in range(n):
            for j in range(m):
                t = testdata[i, ].reshape((self.dim,1))
                a = np.matmul(t, vecs[j])
                exponent = complex(0, a)
                v[i, j] = cmath.exp(exponent)
        pred = np.matmul(v, coeff)
        return pred
