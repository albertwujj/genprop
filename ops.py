import numpy as np
from backprop import Node

def dot(a, b):
    val = a.val.dot(b.val)
    def jacob(grad):
        a_grad = np.zeros(a.val.shape)
        b_grad = np.zeros(b.val.shape)
        for i in range(grad.shape[0]):
            a_grad[i, :] = np.sum(grad[(i,), :] * b.val, axis=1)
        for j in range(grad.shape[1]):
            b_grad[:, j] = np.sum(grad[:, (j,)] * a.val, axis=0)
        return (a_grad, b_grad)
    return Node(val, (a, b), jacob)

def softmax(a):
    def _softmax(x):
        e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), -1))
        return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)
    val = _softmax(a.val)
    # grad not included, can just use derivative of loss with respect to logits directly
    return Node(val, (a, ), None)

def mul(a, b):
    val = a.val * b.val
    jacob = lambda grad: (grad * b.val, grad * a.val)
    return Node(val, (a, b), jacob)

def mean(a, axis=None):
    val = a.val.mean(axis=axis)
    size = a.val.size if axis is None else a.val.shape[axis]
    jacob = lambda grad: (np.full(a.val.shape, grad/size), )
    return Node(val, (a, ), jacob)

def stop_gradient(a):
    return Node(a.val, (a, ), None)
