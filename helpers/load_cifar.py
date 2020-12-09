import numpy as np
from skimage import color


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_cifar10_data():
    ims = [None] * 6
    for i in range(1, 6):
        ims_curr = unpickle(
            f'Data/cifar-10-batches-py/data_batch_{i}')[b'data']
        ims_curr = ims_curr.reshape((10000, 3, 32, 32))
        ims[i-1] = ims_curr

    ims_test = unpickle('Data/cifar-10-batches-py/test_batch')[b'data']
    ims_test = ims_test.reshape((10000, 3, 32, 32))
    ims[5] = ims_test
    final_block = np.concatenate(ims)
    final_block = np.moveaxis(final_block, 1, -1)
    return final_block


def get_cifar100_data():
    ims_train = unpickle('Data/cifar-100-python/train')[b'data']
    ims_train = ims_train.reshape((50000, 3, 32, 32))
    ims_test = unpickle('Data/cifar-100-python/test')[b'data']
    ims_test = ims_test.reshape((10000, 3, 32, 32))
    final_block = np.concatenate((ims_train, ims_test))
    final_block = np.moveaxis(final_block, 1, -1)
    return final_block


if __name__ == '__main__':

    cifar10 = get_cifar10_data()
    cifar100 = get_cifar100_data()
    print(cifar10.shape)
    print(cifar100.shape)
