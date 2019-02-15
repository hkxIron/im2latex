import tensorflow as tf
import collections
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple


AttentionState = collections.namedtuple("AttentionState", ("cell_state", "o"))


class AttentionCell(RNNCell):
    def __init__(self, cell, attention_mechanism, dropout, attn_cell_config,
        num_proj, dtype=tf.float32):
        """
        Args:
            cell: (RNNCell)
            attention_mechanism: (AttentionMechanism)
            dropout: (tf.float)
            attn_cell_config: (dict) hyper params

        """
        # variables and tensors
        self._cell                = cell
        self._attention_mechanism = attention_mechanism
        self._dropout             = dropout

        # hyperparameters and shapes
        self._n_channels     = self._attention_mechanism._n_channels
        self._dim_e          = attn_cell_config["dim_e"]
        self._dim_o          = attn_cell_config["dim_o"]
        self._num_units      = attn_cell_config["num_units"]
        self._dim_embeddings = attn_cell_config["dim_embeddings"]
        self._num_proj       = num_proj
        self._dtype          = dtype

        # for RNNCell
        self._state_size = AttentionState(self._cell._state_size, self._dim_o)


    @property
    def state_size(self):
        return self._state_size # tuple


    @property
    def output_size(self):
        return self._num_proj


    @property
    def output_dtype(self):
        return self._dtype


    def initial_state(self):
        """Returns initial state for the lstm"""
        initial_cell_state = self._attention_mechanism.initial_cell_state(cell=self._cell)
        initial_o          = self._attention_mechanism.initial_state(name="o", dim=self._dim_o)

        return AttentionState(initial_cell_state, initial_o)


    def step(self, embedding, attn_cell_state):
        """
        Args:
            embedding: shape = (batch_size, dim_embeddings) embeddings
                from previous time step
            attn_cell_state: (AttentionState) state from previous time step

        """
        # prev_cell_state: [batch, dim_o]
        # o: [batch, dim_o]
        prev_cell_state, o = attn_cell_state

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # compute new h
            # h_(t) = tanh(Whh * h_(t-1) + Whx * x_t)
            x                     = tf.concat([embedding, o], axis=-1)
            # 此处的call是调用RNNCell的call函数
            # new_h: [batch, num_units]
            new_h, new_cell_state = self._cell.__call__(inputs=x, state=prev_cell_state)
            new_h = tf.nn.dropout(new_h, self._dropout)

            # compute attention
            # new_h: [batch, num_units]
            # context: c: [batch, channel]
            context = self._attention_mechanism.context(new_h)

            # compute o
            # o_W_c: [n_channels, dim_o]
            o_W_c = tf.get_variable("o_W_c", dtype=tf.float32, shape=(self._n_channels, self._dim_o))
            # o_W_h: [num_units, dim_o]
            o_W_h = tf.get_variable("o_W_h", dtype=tf.float32, shape=(self._num_units, self._dim_o))

            # new_o: [batch, dim_o]
            new_o = tf.tanh(tf.matmul(new_h, o_W_h) + tf.matmul(context, o_W_c))
            new_o = tf.nn.dropout(new_o, self._dropout)

            # y_W_o: [dim_o, num_proj=vocab_size]
            y_W_o = tf.get_variable("y_W_o", dtype=tf.float32, shape=(self._dim_o, self._num_proj))
            # logits: [batch, vocab_size]
            logits = tf.matmul(new_o, y_W_o)

            # new Attn cell state
            # new_cell_state:[batch, dim_o]
            # new_o:[batch, dim_o]
            new_state = AttentionState(new_cell_state, new_o) # tuple
            return logits, new_state


    def __call__(self, inputs, state):
        """
        Args:
            inputs: the embedding of the previous word for training only
            state: (AttentionState) (h, o) where h is the hidden state and
                o is the vector used to make the prediction of
                the previous word

        """
        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)