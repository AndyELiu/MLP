from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from mlp import MLPRegressor

tf.logging.set_verbosity(tf.logging.INFO)

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

regressor = MLPRegressor([2, 10, 10, 1], 'relu', learning_rate=0.001, optimizer='Adam')

print(np.sum(x_train))
print(np.sum(z_train))
regressor.fit(x=x_train, y=z_train, steps=100000)

l2_error = regressor.eval(x=x_test, y=z_test)
print('l2 error: {}'.format(l2_error))

#accuracy_score = classifier.eval(x=test_set.data, y=test_set.target)
#print('Accuracy: {}'.format(accuracy_score))

#new_samples = np.array(
#    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
#y = classifier.predict(new_samples)
#print('Predictions: {}'.format(str(y)))
