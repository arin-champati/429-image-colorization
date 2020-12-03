from pytorch2keras import pytorch_to_keras
from pytorch_model import eccv16
from base_color import BaseColor
import keras
import torch
import numpy as np
from torch.autograd import Variable

model = eccv16()
input_np = np.random.uniform(0, 1, (1, 256, 256, 1))
input_var = Variable(torch.FloatTensor(input_np))
input_shape = []
k_model = pytorch_to_keras(model, input_var, verbose=True)
print(k_model.summary())
