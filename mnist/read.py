import numpy as np
import struct

def read_mnist():
    def read_idx(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    fnames = ('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')
    return (read_idx('mnist/'+fname) for fname in fnames)