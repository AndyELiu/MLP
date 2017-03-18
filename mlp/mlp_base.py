from abc import ABC, abstractmethod
import tensorflow as tf
import math


class MLPBase(ABC):
    def __init__(self, network_shape, activation='relu', learning_rate=0.001):
        self.network_shape = network_shape
        self.activation = activation
        self.learning_rate = learning_rate
        self.x, self.y, self.logits, self.loss = (None,) * 4
        self.precision, self.prediction = (None,) * 2
        self.sess = tf.Session()

        assert self.activation in ('tanh', 'sigmoid', 'relu', 'crelu')
        self._build_network()

    def _build_network(self):
        self._build_placeholder()
        x = self.x
        for layer_index in range(1, len(self.network_shape)):
            x = self._build_layer(layer_index, x)

        self.logits = x
        self._build_loss()
        self._build_train_op()
        self._build_eval_op()
        self._build_predict_op()
        self.sess.run(tf.global_variables_initializer())

    #  two placeholders, self.x and self.y
    @abstractmethod
    def _build_placeholder(self):
        pass

    #  build a layer
    def _build_layer(self, layer_index, inputs):
        n_in = self.network_shape[layer_index - 1]
        n_out = self.network_shape[layer_index]
        is_final_layer = (layer_index == len(self.network_shape) - 1)

        w = tf.Variable(tf.random_normal([n_in, n_out],
                        stddev=math.sqrt(1.0/n_in)),
                        name='w{}'.format(layer_index))
        b = tf.Variable(tf.zeros([n_out]), name='b{}'.format(layer_index))

        outputs = tf.matmul(inputs, w) + b
        if not is_final_layer:
            if self.activation == 'tanh':
                outputs = tf.nn.tanh(outputs)
            elif self.activation == 'sigmoid':
                outputs = tf.nn.sigmoid(outputs)
            elif self.activation == 'relu':
                outputs = tf.nn.relu(outputs)
            else:
                raise Exception('invalid activation!')

        return outputs

    @abstractmethod
    def _build_loss(self):
        pass

    def _build_train_op(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    @abstractmethod
    def _build_eval_op(self):
        pass

    @abstractmethod
    def _build_predict_op(self):
        pass

    def fit(self, x, y, steps=10000, report_per_step=100):
        for i in range(steps):
            feed_dict = {self.x: x, self.y: y}
            _, loss = self.sess.run([self.train_op, self.loss],
                                    feed_dict=feed_dict)
            if i % report_per_step == 0:
                print("steps: {}, loss: {}".format(i, loss))

    def eval(self, x, y):
        feed_dict = {self.x: x, self.y: y}
        precision = self.sess.run([self.precision], feed_dict=feed_dict)
        return precision

    def predict(self, x):
        feed_dict = {self.x: x}
        prediction = self.sess.run([self.prediction], feed_dict=feed_dict)
        return prediction
