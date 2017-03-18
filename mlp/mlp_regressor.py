from .mlp_base import MLPBase
import tensorflow as tf


class MLPRegressor(MLPBase):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)

    def _build_placeholder(self):
        self.x = tf.placeholder(tf.float32, [None, self.network_shape[0]],
                                name='x')
        self.y = tf.placeholder(tf.float32, [None, self.network_shape[-1]],
                                name='y')

    def _build_loss(self):
        self.loss = tf.nn.l2_loss(self.logits - self.y)

    def _build_eval_op(self):
        self.precision = tf.nn.l2_loss(self.logits - self.y)

    def _build_predict_op(self):
        self.prediction = self.logits
