'''
    Generate
    1 2 1 1 1 1 2 1 1 1 1 2 1
    2 2 2 2 2 2 2 2 2 2 2 2 2
    1 2 1 1 1 1 2 1 1 1 1 2 1
    1 2 1 3 3 1 2 1 3 3 1 2 1
    1 2 1 3 3 1 2 1 3 3 1 2 1
    1 2 1 1 1 1 2 1 1 1 1 2 1
    2 2 2 2 2 2 2 2 2 2 2 2 2
    1 2 1 1 1 1 2 1 1 1 1 2 1
    1 2 1 3 3 1 2 1 3 3 1 2 1
    1 2 1 3 3 1 2 1 3 3 1 2 1
    1 2 1 1 1 1 2 1 1 1 1 2 1
    2 2 2 2 2 2 2 2 2 2 2 2 2
    1 2 1 1 1 1 2 1 1 1 1 2 1

    Investigate differences between softmax and sigmoid functions
'''

# libraries
import torch as th
import numpy as np


def sigmoid(x):
    return 1/(1+th.exp(-x))

def softmax(x):
    return th.exp(x)/th.exp(x).sum()


if __name__ == '__main__':

    x = th.full((13,13), 1, dtype=th.int)
    x[:,[1,6,11]] = 2
    x[[1,6,11],:] = 2
    x[3:5,3:5] = 3
    x[8:10,3:5] = 3
    x[3:5,8:10] = 3
    x[8:10,8:10] = 3
    print(x)


    x = th.randn((3,1))
    sigmoid(x)
    softmax(x)
    softmax(x).sum()

    # Making predictions
    y = np.array([0,1,0]) # multi-class
    y = np.array([0,1,1]) # multi-label

    x = th.tensor([-3.,2,3,7])
    sigmoid(x)
    np.round(softmax(x),2)
