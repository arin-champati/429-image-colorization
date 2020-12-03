import numpy as np
import keras.backend as K
from keras.losses import MSE
from helpers.utils import annealed_mean, get_qab

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


def regression_loss_color(y_true, y_pred):
    image_pred = annealed_mean(y_pred)
    mse = MSE(y_true, image_pred)
    return K.mean(mse)


def lab_loss(y_true, y_pred):
    gt = get_qab(y_true)
    alpha = 0.1

    # regression loss
    rl = regression_loss_color(y_true, y_pred)

    # output loss
    final_loss = cl + alpha * rl

    return final_loss
