import numpy as np
from mnist.read import read_mnist
from backprop import Node
from ops import dot, sigmoid, sub, power, mean

# preprocess
x_train, labels_train, x_test, labels_test = read_mnist()
y_train = np.zeros((labels_train.shape[0], 10))
y_train[np.arange(labels_train.shape[0]), labels_train] = 1
x_test = np.reshape(x_test, (x_test.shape[0], -1))
x_train = np.reshape(x_train, (x_train.shape[0], -1))

layer_size = 800
w1 = Node(val=np.random.rand(x_train.shape[1],layer_size), changeable=True)
w2 = Node(val=np.random.rand(layer_size,10), changeable=True)

def neural_net(x):
    h = x
    for w in (w1, w2):
        h = dot(h, w)
    return sigmoid(h)

batch_size = 10
for bi in range(0, 3000, batch_size):

    end = min(x_train.shape[0], bi+batch_size)
    x = Node(val=x_train[bi:end], changeable=False)
    labels = Node(val=y_train[bi:end], changeable=False)

    h = neural_net(x)
    loss = mean(power(sub(h, labels)), axis=0)
    loss.backprop(np.asarray([-.0001]*10))

    if bi % 1000 == 0:
        print('1000 trained')

preds = neural_net(Node(val=x_test, changeable=False))
accuracy = np.count_nonzero(np.argmax(preds, axis=-1)==labels_test)//len(labels_test)
print(accuracy)