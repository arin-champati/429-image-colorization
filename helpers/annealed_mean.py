import keras.backend as K
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.python.keras import backend
from load_cifar import get_cifar100_data
from skimage import color

gamut = np.load('Data/pts_in_hull.npy', allow_pickle=True)
nn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(gamut)
gamut_tensor = K.constant(gamut)


def annealed_softmax(qab, temperature=0.38):
    expon = K.pow(qab, 1/temperature)
    expon /= K.sum(expon, axis=-1, keepdims=True)
    return expon


def annealed_mean(qab, temperature=0.38):
    qab = annealed_softmax(qab, temperature)
    am = K.dot(qab, gamut_tensor)
    return am


def get_qab(img_ab, sigma=5, bins=313):
    # gets num_nb nearest neighbors
    batch, h, w, _ = img_ab.shape
    a = np.ravel(img_ab[:, :, :, 0])
    b = np.ravel(img_ab[:, :, :, 1])
    ab = np.vstack((a, b)).T  # now dim (H*W*2)
    distances, idx = nn.kneighbors(ab)

    # smoothen
    gaussian = np.exp(-distances**2 / (2*sigma**2))
    gaussian /= np.expand_dims(np.sum(gaussian, axis=1), axis=-1)
    soft_encoding = np.zeros((ab.shape[0], bins))
    pts = np.expand_dims(np.arange(h*w), axis=-1)
    soft_encoding[pts, idx] = gaussian
    return K.constant(soft_encoding.reshape(batch, h, w, bins))


def get_qab1(img_ab, sigma=5, bins=313):
    # gets num_nb nearest neighbors
    batch, h, w, _ = img_ab.shape
    a = np.ravel(img_ab[:, :, :, 0])
    b = np.ravel(img_ab[:, :, :, 1])
    ab = np.vstack((a, b)).T  # now dim (H*W*2)
    distances, idx = nn.kneighbors(ab)

    # smoothen
    gaussian = np.exp(-distances**2 / (2*sigma**2))
    gaussian /= np.expand_dims(np.sum(gaussian, axis=1), axis=-1)
    flattened = np.dstack((idx, gaussian))
    output = np.reshape(flattened, (batch, h, w, sigma, 2))
    return K.constant(output)


ab = np.expand_dims(get_cifar100_data()[0], axis=0)
ab = color.rgb2lab(ab)[:, :, :, 1:]
qab = get_qab(ab)
qab1 = get_qab1(ab)
idx = K.cast(qab1[:, :, :, :, 0], 'int32')
idx = K.reshape(idx, (-1, 5))
qab = K.reshape(qab, (-1, 313))
pts = K.expand_dims(K.arange(1024), axis=-1)
qab2 = qab[pts, idx]

print(qab.shape)
print(idx.shape)
print(qab2.shape)

"""
a = np.array([[4, 3, 2, 4, 1], [9, 9, 4, 6, 2]])
b = np.array([[1, 0], [0, 1]])
pts = np.expand_dims(np.arange(2), axis=-1)
print(a[pts, b])
"""
