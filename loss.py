import numpy as np
import keras.backend as K
from helpers.utils import annealed_mean, get_qab

prior_factor = np.load("data/prior_probs.npy").astype(np.float32)


def categorical_crossentropy_color(y_true, y_pred):

    q = 313
    y_true = K.reshape(y_true, (-1, q))
    y_pred = K.reshape(y_pred, (-1, q))

    idx_max = K.argmax(y_true, axis=1)
    weights = K.gather(prior_factor, idx_max)
    weights = K.reshape(weights, (-1, 1))

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = K.categorical_crossentropy(y_pred, y_true)
    cross_ent = K.mean(cross_ent, axis=-1)

    return cross_ent


def regression_loss_color(y_true, y_pred):
    actual_image = annealed_mean(y_pred)
    mse = K.mean(K.square(actual_image - y_true), axis=-1)
    return mse


def loss(y_true, y_pred):
    gt = get_qab(y_true)
    alpha = 0.5

    # cross-entropy
    categorical_crossentropy_color(gt, y_pred)

    # regression loss
    regression_loss_color(y_true, y_pred)

    final_loss = categorical_crossentropy_color + \
        (alpha * regression_loss_color)

    return final_loss
