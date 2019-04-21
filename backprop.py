class Node():
    def __init__(self, val, inputs=None, jacob=None, changeable=False):
        self.inputs = inputs
        self.jacob = jacob
        self.val = val
        self.changeable = changeable

    def backprop(self, grad):
        if self.jacob:
            input_grads = self.jacob(grad)
            for input, input_grad in zip(self.inputs, input_grads):
                input.backprop(input_grad)

        if self.changeable:
            self.val += grad