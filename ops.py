import numpy as np

from backprop import Node

def sub(a, b):
    val = a.val - b.val
    jacob = lambda dy: (np.full(a.val.shape, dy), -np.full(a.val.shape,dy))
    return Node(val, (a, b), jacob)

def power(a):
    val = a.val**2
    jacob = lambda dy: (np.abs(dy * 2 * a.val),)
    return Node(val, (a,), jacob)

def sigmoid(a):
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    val = _sigmoid(a.val)
    jacob = lambda dy: (dy * _sigmoid(a.val) * (1-_sigmoid(a.val)),)
    return Node(val, (a,), jacob)


def dot(a, b):
    val = a.val.dot(b.val)
    def jacob(dy):
        a_grad = np.zeros(a.val.shape)
        b_grad = np.zeros(b.val.shape)
        for i in range(dy.shape[0]):
            a_grad[i, :] = np.sum(dy[(i,),:] * b_grad, axis=-1)
        for j in range(dy.shape[1]):
            b_grad[:, j] = np.sum(dy[:,(j,)] * a_grad, axis=-2)
        return a_grad, b_grad
    return Node(val, (a,b), jacob)


def dotND(a, b):
    val = a.val.dot(b.val)
    def jacob(dy):
        a_grad = np.zeros(a.val.shape if len(a.val.shape) > 2 else (1,) + a.val.shape)
        b_grad = np.zeros(a.val.shape if len(b.val.shape) > 2 else (1,) + b.val.shape)
        for index in np.ndindex(*(a_grad.shape[:-2] + b_grad.shape[:-2])):
            for i in range(dy.shape[0]):
                a_grad[index, i, :] = np.sum(dy[index, (i,),:] * b_grad[index], axis=-1)
            for j in range(dy.shape[1]):
                b_grad[index, :, j] =  np.sum(dy[index, :,(j,)] * a_grad[index], axis=-2)
        return a_grad if len(a.val.shape) > 2 else a_grad.squeeze(0), \
               b_grad if len(b.val.shape) > 2 else b_grad.squeeze(0)

    return Node(val, (a, b), jacob)

def mean(a, axis=0):
    val = a.val.mean(axis=axis)
    size = a.val.shape[axis]
    jacob = lambda dy: (np.repeat(np.expand_dims(dy/size, axis=axis), size, axis=axis),)
    return Node(val, (a,), jacob)


