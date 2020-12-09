import numpy as np
import keras.backend as K
from keras.losses import MSE
from helpers.annealed_mean import annealed_mean, get_qab

prior_probs = np.load("data/prior_probs.npy").astype(np.float32)


def categorical_crossentropy_color(y_true, y_pred):

    q = 313
    y_true = K.reshape(y_true, (-1, q))
    y_pred = K.reshape(y_pred, (-1, q))

    idx_max = K.argmax(y_true, axis=1)
    weights = K.gather(prior_probs, idx_max)
    weights = K.reshape(weights, (-1, 1))

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = K.categorical_crossentropy(y_pred, y_true)
    cross_ent = K.mean(cross_ent, axis=-1)

    return cross_ent


def categorical_crossentropy_1hot(y_true, y_pred):
    q = 313
    y_true = K.reshape(y_true, (-1, 1))
    y_pred = K.reshape(y_pred, (-1, q))
    weights = K.gather(prior_probs, y_true)
    weights = K.reshape(weights, (-1, 1))
    y_pred *= weights
    loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    return loss


def regression_loss_color(y_true, y_pred):
    image_pred = annealed_mean(y_pred)
    return K.mean(K.square(image_pred-y_true))


def lab_loss(y_true, y_pred):
    # regression loss
    rl = regression_loss_color(y_true, y_pred)

    return 0.1 * rl
