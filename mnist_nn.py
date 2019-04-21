import numpy as np
from mnist.read import read_mnist
from backprop import Node
from ops import dot, softmax, sub, power, mean

# preprocess
datas = list(read_mnist())
[np.random.shuffle(arr) for arr in datas]
x_train, labels_train, x_test, labels_test = datas
y_train = np.zeros((labels_train.shape[0], 10))
y_train[np.arange(labels_train.shape[0]), labels_train] = 1
x_test = np.reshape(x_test, (x_test.shape[0], -1))
x_train = np.reshape(x_train, (x_train.shape[0], -1))

layer_size = 800
w1 = Node(val=(np.random.rand(x_train.shape[1],layer_size) - .5)*.01, changeable=True)
w2 = Node(val=(np.random.rand(layer_size,10) - .5)*.01, changeable=True)

def neural_net(x):
    h = x
    for w in (w1, w2):
        h = dot(h, w)
    return h, softmax(h)

batch_size = 50
for bi in range(0, 60000, batch_size):

    end = min(x_train.shape[0], bi+batch_size)
    x = Node(val=x_train[bi:end], changeable=False)
    y = y_train[bi:end]

    logits, pred = neural_net(x)

    logits.backprop(y-pred.val*.0001/y.shape[0])

    if bi % 1000 == 0:

        print('1000 trained')

print('testing')
_, preds = neural_net(Node(val=x_train[:10000], changeable=False))
print(preds.val)
test_accuracy = np.count_nonzero(np.argmax(preds.val, axis=-1)==labels_train[:10000])/10000
print(test_accuracy)