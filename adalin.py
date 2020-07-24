# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:13:40 2016

@author: kurita
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# learning rate
alpha = 0.003

# read iris data from file 
iris = np.loadtxt('iris.dat')
nr, nc = iris.shape

# feature vectors of each class
#
D = iris[0:100,:]
T = np.zeros(100)
T[0:50] = 1

#print(nr, nc)

#print(D)
#print(T)

# initialize the weights of the perceptron
#
W = (np.random.rand(nc)-0.5) / 1000.0
h = ((np.random.rand(1) - 0.5) / 1000.0)[0]
YY = np.zeros(100)
EE = np.zeros(100)

fig = plt.figure()
while True:
    ind = np.random.randint(0, high=100)
    x = D[ind,:].T
    t = T[ind]
    # output of the perceptron
    y = np.dot(W.T,x) - h
    # update the parameters
    W = W + alpha * (t-y) * x
    h = h - alpha * (t-y)

    print('W', W)
    print('h', h)

    # plot
    EE = np.dot(D, W) - h
    YY[EE >= 0.5] = 1
    YY[EE < 0.5] = 0
#    xrange = np.arange(1,100)
#    plt.plot(xrange, EE)

    plt.plot(EE)
    plt.plot(YY)
#    plt.show(block=False)
    plt.axis((0,100,-0.2, 1.2))
    plt.pause(0.01)

    # key input
    print('Please input anythin (q for quit)')
    input_line = sys.stdin.readline()
    plt.clf()
#    plt.close(fig)
#    print input_line2 + " is writen"
#    input_chars = raw_input('>>>  ')
#    print 'input word is ', input_chars
    
    if (input_line == 'q\n'):
        plt.close(fig)
        break


