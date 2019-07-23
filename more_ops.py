import numpy as np
from backprop import Node

def dotND(a, b):
    val = a.val.dot(b.val)
    def jacob(grad):
        a_val = a.val
        b_val = b.val
        if len(a_val.shape) < 3:
            np.expand_dims(a_val, -1)
        if len(b_val.shape) < 3:
            np.expand_dims(b_val, -1)

        a_grad = np.zeros(a_val.shape)
        b_grad = np.zeros(b_val.shape)
        for index in np.ndindex(*(a_grad.shape[:-2] + b_grad.shape[:-2])):
            for i in range(grad.shape[0]):
                a_grad[index, i, :] = np.sum(grad[index, (i,),:] * b_val[index], axis=-1)
            for j in range(grad.shape[1]):
                b_grad[index, :, j] =  np.sum(grad[index, :,(j,)] * a_val[index], axis=-2)
        return np.reshape(a_grad, a.val.shape), np.reshape(b_grad, b.val.shape)

    return Node(val, (a, b), jacob)

# shape ops

def slice(a, slice):
    val = a.val[slice]
    def jacob(grad):
        a_grad = np.zeros(a.val.shape)
        a_grad[slice] = grad
        return (a_grad, )
    return Node(val, (a, ), jacob)

def concat(inputs, axis=0):
    val = np.concatenate([x.val for x in inputs], axis=axis)
    sizes = np.cumsum([x.val.shape[axis] for x in inputs[1:]])
    def jacob(grad):
        return tuple(np.split(grad, sizes, axis=axis))
    return Node(val, inputs, jacob)

