import keras.backend as K
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors

gamut = np.load('Data/pts_in_hull.npy', allow_pickle=True)
nn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(gamut)
gamut_tensor = K.constant(gamut)


def annealed_softmax(qab, temperature=0.38):
    expon = K.pow(qab, 1/temperature)
    expon /= K.sum(expon, axis=-1, keepdims=True)
    return expon


def annealed_mean(qab, temperature=1):
    qab = annealed_softmax(qab, temperature)
    am = K.dot(qab, gamut_tensor)
    return am


def get_qab(img_ab, sigma=5, bins=313):

    # gets num_nb nearest neighbors
    h, w, _ = img_ab.shape
    a = np.ravel(img_ab[:, :, 0])
    b = np.ravel(img_ab[:, :, 1])
    ab = np.vstack((a, b)).T  # now dim (H*W*2)
    distances, idx = nn.kneighbors(ab)

    # smoothen
    gaussian = np.exp(-distances**2 / (2*sigma**2))
    gaussian /= np.expand_dims(np.sum(gaussian, axis=1), axis=-1)
    soft_encoding = np.zeros((ab.shape[0], bins))
    pts = np.expand_dims(np.arange(h*w), axis=-1)
    soft_encoding[pts, idx] = gaussian
    return K.constant(soft_encoding.reshape(h, w, bins))
