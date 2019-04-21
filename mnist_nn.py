import numpy as np
from mnist.preprocessing import get_mnist
from backprop import Node
from ops import dot, softmax

n_classes = 5
x_train, y_train, x_test, y_test = get_mnist(n_classes)
x_train, x_test = (np.reshape(x, (x.shape[0], -1)) for x in (x_train, x_test)) # flatten
x_train, x_test = (np.concatenate([x, np.ones((x.shape[0], 1))], axis=1) for x in (x_train, x_test)) # add bias elt

layer_size = 500
w1 = Node(val=np.random.randn(x_train.shape[1], layer_size) / 10, changeable=True)
w2 = Node(val=np.random.randn(layer_size, n_classes) / 10, changeable=True)

def neural_net(x):
    h = x
    for w in (w1, w2):
        h = dot(h, w)
    return h, softmax(h)

batch_size = 32
lr = .0003
for epoch in range(1):
    for bi in range(0, x_train.shape[0], batch_size):
        end = min(x_train.shape[0], bi + batch_size)
        x = Node(val=x_train[bi:end])
        y = y_train[bi:end]

        logits, pred = neural_net(x)
        logits.backprop(lr * (y - pred.val))

_, preds = neural_net(Node(val=x_test))
accuracy = np.count_nonzero(np.argmax(preds.val, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print(accuracy)
