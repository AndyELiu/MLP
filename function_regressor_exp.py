from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from mlp import MLPRegressor

# Data sets
N = 100
alpha = 0.8
num_train = int(np.ceil(alpha*N*N))

x = np.linspace(-10, 10, N)
y = np.linspace(-10, 10, N)
xc, yc = np.meshgrid(x, y)
zc = xc * yc

z = np.stack([xc, yc], 0)
z = np.transpose(np.reshape(z, (2, N*N)))
zc = np.reshape(zc, (N*N,))

zzc = np.concatenate([z, zc[:, np.newaxis]], axis=1)
np.random.shuffle(zzc)
zzc_train = zzc[:num_train, :]
zzc_test = zzc[num_train:, :]

x_train = zzc_train[:, :2]
z_train = zzc_train[:, 2:]

x_test = zzc_test[:, :2]
z_test = zzc_test[:, 2:]

# Create regressor
regressor = MLPRegressor([2, 10, 10, 1], 'relu', learning_rate=0.001,
                         optimizer='Adam')

# Fitting
regressor.fit(x=x_train, y=z_train, steps=100000)

# Evaluation
l2_error = regressor.eval(x=x_test, y=z_test)
print('l2 error: {}'.format(l2_error))
