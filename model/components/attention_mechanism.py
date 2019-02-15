import tensorflow as tf


class AttentionMechanism(object):
    """Class to compute attention over an image"""

    def __init__(self, img, dim_e, tiles=1):
        """Stores the image under the right shape.

        We loose the H, W dimensions and merge them into a single
        dimension that corresponds to "regions" of the image.

        Args:
            img: (tf.Tensor) image, [batch, heigth, weight, channel]
            dim_e: (int) dimension of the intermediary vector used to
                compute attention
            tiles: (int) default 1, input to context h may have size
                    (tile * batch_size, ...)

        """
        if len(img.shape) == 3:
            self._img = img
        elif len(img.shape) == 4:
            N    = tf.shape(img)[0]
            H, W = tf.shape(img)[1], tf.shape(img)[2] # image
            C    = img.shape[3].value                 # channels
            self._img = tf.reshape(img, shape=[N, H*W, C])
        else:
            print("Image shape not supported")
            raise NotImplementedError

        # dimensions
        # _img:[N, HW, C]
        self._n_regions  = tf.shape(self._img)[1] # HW
        self._n_channels = self._img.shape[2].value # C
        self._dim_e      = dim_e # 512
        self._tiles      = tiles
        self._scope_name = "att_mechanism"

        # attention vector over the image
        # _att_img = activation(inputs * W + bias)
        # _img:[N, HW, C], W:[C, dim_e], 只会对img的最后一维进行乘法
        # _att_img: [N, HW, dim_e], 将原始图像变为seq2seq中常见的格式,即类似于[batch, sequence_length, embedding_size]
        self._att_img = tf.layers.dense(
            inputs=self._img,
            units=self._dim_e,
            activation=None,
            use_bias=False,
            name="att_img")


    def context(self, h):
        """Computes attention

        Args:
            h: (batch_size, num_units) hidden state

        Returns:
            context: (batch_size, channels) context vector

        """
        with tf.variable_scope(self._scope_name):
            if self._tiles > 1:
                # _att_img:[N, H*W, dim_e], att_img:[N, 1, H*W, dim_e]
                att_img = tf.expand_dims(self._att_img, axis=1)
                # att_img: [N, tiles, H*W, dim_e]
                att_img = tf.tile(att_img, multiples=[1, self._tiles, 1, 1])
                # att_img: [N*tiles, H*W=_n_regions, dim_e], H*W = regions
                att_img = tf.reshape(att_img, shape=[-1, self._n_regions, self._dim_e])
                # _img:[N, H*W, C], img:[N, 1, H*W, C]
                img = tf.expand_dims(self._img, axis=1)
                # img:[N, tiles, H*W, C]
                img = tf.tile(img, multiples=[1, self._tiles, 1, 1])
                # img:[N*tiles, H*W=regions, C]
                img = tf.reshape(img, shape=[-1, self._n_regions, self._n_channels])
            else:
                #_att_img: [N, H*W, dim_e]
                att_img = self._att_img
                # img:[N, H*W, C]
                img     = self._img

            # computes attention over the hidden vector
            # h:[batch, num_units], W:[num_units, _dim_e]
            # att_h:[batch, _dim_e]
            att_h = tf.layers.dense(inputs=h, units=self._dim_e, activation=False, use_bias=False)

            # sums the two contributions
            # att_h:[batch, 1, dim_e]
            att_h = tf.expand_dims(att_h, axis=1)
            # att: [batch, N*W, dim_e]
            att = tf.tanh(att_img + att_h)

            # computes scalar product with beta vector
            # works faster with a matmul than with a * and a tf.reduce_sum
            # att_beta: [dim_e, 1]
            att_beta = tf.get_variable("att_beta", shape=[self._dim_e, 1], dtype=tf.float32)
            # att_flat:[batch*N*W, dim_e]
            att_flat = tf.reshape(att, shape=[-1, self._dim_e])
            # e:[batch*N*W, 1]
            e = tf.matmul(att_flat, att_beta)
            # e:[batch, N*W]
            e = tf.reshape(e, shape=[-1, self._n_regions])

            # compute weights
            # attention_weight:[batch, H*W]
            attention_weight = tf.nn.softmax(e, axis=-1)
            # attention_weight:[batch=N, H*W, 1]
            attention_weight = tf.expand_dims(attention_weight, axis=-1)
            # img:[N, H*W=regions, C]
            # context: [N, C]
            context = tf.reduce_sum(attention_weight * img, axis=1)

            return context


    def initial_cell_state(self, cell):
        """Returns initial state of a cell computed from the image

        Assumes cell.state_type is an instance of named_tuple.
        Ex: LSTMStateTuple

        Args:
            cell: (instance of RNNCell) must define _state_size

        """
        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            # h: [N, hidden_dim]
            h = self.initial_state(hidden_name, hidden_dim)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell


    def initial_state(self, name, dim):
        """Returns initial state of dimension specified by dim"""
        with tf.variable_scope(self._scope_name):
            # _img:[N, HW, C]
            # img_mean: [N, C]
            img_mean = tf.reduce_mean(self._img, axis=1)
            # W:[C, hidden_dim]
            W = tf.get_variable(name="W_{}_0".format(name), shape=[self._n_channels, dim])
            # b:[hidden_dim]
            b = tf.get_variable(name="b_{}_0".format(name), shape=[dim])
            # h: [N, hidden_dim]
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h