import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell


from .components.positional import add_timing_signal_nd


class Encoder(object):
    """Class with a __call__ method that applies convolutions to an image"""

    def __init__(self, config):
        self._config = config


    def __call__(self, training, img, dropout):
        """Applies convolutions to the image

        Args:
            training: (tf.placeholder) tf.bool
            img: batch of img, shape = (?, height, width, channels), of type
                tf.uint8

        Returns:
            the encoded images, shape = (?, h', w', c')

        """

        # img: [batch, height, width, channels=1]
        img = tf.cast(img, tf.float32) / 255.

        with tf.variable_scope("convolutional_encoder"):
            # conv + max pool -> /2
            # out:[batch, height, width, 64]
            out = tf.layers.conv2d(inputs=img, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
            # out:[batch, height/2, width/2, 64]
            out = tf.layers.max_pooling2d(inputs=out, pool_size=2, strides=2, padding="SAME")

            # conv + max pool -> /2
            # out:[batch, height/2, width/2, 128]
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME", activation=tf.nn.relu)
            # out:[batch, height/4, width/4, 128]
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            # regular conv -> id
            # out:[batch, height/4, width/4, 256]
            out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)
            # out:[batch, height/4, width/4, 256]
            out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)

            if self._config.encoder_cnn == "vanilla":
                out = tf.layers.max_pooling2d(out, pool_size=(2, 1), strides=(2, 1), padding="SAME")

            # out:[batch, height/4, width/4, 512]
            out = tf.layers.conv2d(out, 512, 3, 1, "SAME", activation=tf.nn.relu)

            if self._config.encoder_cnn == "vanilla":
                out = tf.layers.max_pooling2d(out, pool_size=(1, 2), strides=(1, 2), padding="SAME")

            if self._config.encoder_cnn == "cnn":
                # conv with stride /2 (replaces the 2 max pool)
                out = tf.layers.conv2d(out, filters=512, kernal_size=(2, 4), strides=2, padding="SAME")

            # conv
            out = tf.layers.conv2d(out, 512, 3, 1, "VALID", activation=tf.nn.relu)

            if self._config.positional_embeddings:
                # from tensor2tensor lib - positional embeddings
                # 注意:position_embedding直接加在原来的x上
                out = add_timing_signal_nd(out)

        return out
