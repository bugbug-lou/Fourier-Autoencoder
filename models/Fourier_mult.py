import cmath
from scipy.fft import fft, ifft
import numpy as np
import datetime
import math
import random
from matplotlib import pyplot as plt
import multiprocessing
import torch
from torch.autograd import Variable
from torch import optim
from lempel_ziv_complexity import lempel_ziv_complexity
import collections
import argparse


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
    def __init__(self, dim, thres, intl, train_inputs, test_inputs, labels):
        self.inputs = train_inputs
        self.testdata = test_inputs
        self.labels = labels
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
        for v in vecs:
            vecs1 = vecs1 + get_all(v)
        return vecs1

    def get_mat_inv(self):
        inputs, dim, thres = self.inputs, self.dim, 2 * self.thres + 1
        vecs = self.get_axis()
        l = inputs.shape[0]
        num_row = (thres) ** dim
        m = min(l, num_row)
        matrix = np.zeros([m, m], dtype=complex)
        for i in range(m):
            for j in range(m):
                h = inputs[i,].reshape((dim, 1))
                a = np.matmul(h, vecs[j])
                exponent = complex(0, a) / self.intl
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
                t = testdata[i,].reshape((self.dim, 1))
                a = np.matmul(t, vecs[j])
                exponent = complex(0, a) / self.intl
                v[i, j] = cmath.exp(exponent)
        pred = np.matmul(v, coeff)
        return pred
