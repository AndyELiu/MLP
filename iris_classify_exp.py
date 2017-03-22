from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from mlp import MLPClassifier

# Data sets
IRIS_TRAINING = "data/iris_training.csv"
IRIS_TEST = "data/iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
                               filename=IRIS_TRAINING,
                               target_dtype=np.int,
                               features_dtype=np.float32
                               )

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
                               filename=IRIS_TEST,
                               target_dtype=np.int,
                               features_dtype=np.float32
                               )

# Create classifier
classifier = MLPClassifier([4, 10, 10, 3], 'relu')

# Fitting
classifier.fit(x=training_set.data, y=training_set.target)

# Evaluation
accuracy_score = classifier.eval(x=test_set.data, y=test_set.target)
print('Accuracy: {}'.format(accuracy_score))

new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))
