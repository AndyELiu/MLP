from .mlp_base import MLPBase
import tensorflow as tf


class MLPClassifier(MLPBase):
    def __init__(self, *argc, **argv):
        super().__init__(*argc, **argv)

    def _build_placeholder(self):
        self.x = tf.placeholder(tf.float32, [None, self.network_shape[0]],
                                name='x')
        self.y = tf.placeholder(tf.int32, [None], name='label')

    def _build_loss(self):
        scores = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.y
        )

        self.loss = tf.reduce_mean(scores)

    def _build_eval_op(self):
        predits = tf.equal(tf.argmax(self.logits, axis=1),
                           tf.cast(self.y, tf.int64))
        corrects = tf.reduce_sum(tf.cast(predits, tf.int32))
        self.precision = corrects/tf.shape(predits)[0]

    def _build_predict_op(self):
        self.prediction = tf.argmax(self.logits, axis=1)
