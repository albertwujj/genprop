import numpy as np
import struct
import gzip
import os

DATA_URL = 'http://yann.lecun.com/exdb/mnist/'

def download(filename):
    if not os.path.exists('mnist/data/' + filename):
        from urllib.request import urlretrieve
        print("Downloading %s." % DATA_URL + filename + '.gz')
        urlretrieve(DATA_URL + filename + '.gz', 'mnist/data/' + filename)

def read_mnist():
    def read_idx(filename):
        download(filename)
        with gzip.open('mnist/data/' + filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape).copy()

    fnames = ('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')
    return (read_idx(fname) for fname in fnames) # returns the 4 ndarrays


def get_mnist(n_classes=10):

    def select_classes(x, labels):
        inds = np.where(labels < n_classes)
        return x[inds], labels[inds]
    x_train, labels_train, x_test, labels_test = read_mnist()
    (x_train, labels_train), (x_test, labels_test) = (select_classes(x_train, labels_train), select_classes(x_test, labels_test))

    def one_hot(x):
        ret = np.zeros((x.shape[0], n_classes))
        ret[np.arange(x.shape[0]), x] = 1
        return ret

    return x_train/255, one_hot(labels_train), x_test/255, one_hot(labels_test)