from model_paper import build_model
from loss import loss
from config import *
import tensorflow as tf
import keras.backend as K
import numpy as np


def get_training_data(data_path):
    train = np.load(data_path)
    train_examples = train['arr_0']
    train_labels = train['arr_1']
    return train_examples, train_labels


def train():
    model = build_model()
    model.compile(optimizer='adam', loss=loss, metrics=[
                  tf.keras.metrics.MeanSquaredError()])
    print(model.summary())

    train_examples, train_labels = get_training_data(
        'train_25.npz')
    train_data = tf.data.Dataset.from_tensor_slices(
        (train_examples, train_labels))
    train_data = train_data.shuffle(
        20).batch(2)
    epochs = 10

    history = model.fit(train_data, epochs=epochs)


train()
