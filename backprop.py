import numpy as np

class Node():
    def __init__(self, val, inputs=None, jacob = None, changeable=False):
        self.inputs = inputs
        self.jacob = jacob
        self.val = val
        self.changeable = changeable

    def backprop(self, grad):
        if self.jacob:
            grads = self.jacob(grad)
            for input, grad_i in zip(self.inputs, grads):
                input.backprop(grad_i)

        if self.changeable:
            self.val += grad
