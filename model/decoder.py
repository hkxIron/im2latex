import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import GRUCell, LSTMCell


from .components.dynamic_decode import dynamic_decode
from .components.attention_mechanism import AttentionMechanism
from .components.attention_cell import AttentionCell
from .components.greedy_decoder_cell import GreedyDecoderCell
from .components.beam_search_decoder_cell import BeamSearchDecoderCell

# 超赞的代码
class Decoder(object):
    """Implements this paper https://arxiv.org/pdf/1609.04938.pdf"""

    def __init__(self, config, n_tok, id_end):
        self._config = config
        self._n_tok = n_tok
        self._id_end = id_end
        self._tiles = 1 if config.decoding == "greedy" else config.beam_size


    def __call__(self, training, img, formula, dropout):
        """Decodes an image into a sequence of token

        Args:
            training: (tf.placeholder) bool
            img: encoded image (tf.Tensor) shape = (N, H, W, C)
            formula: (tf.placeholder), shape = (N, T)

        Returns:
            pred_train: (tf.Tensor), shape = (?, ?, vocab_size) logits of each class
            pret_test: (structure)
                - pred.test.logits, same as pred_train
                - pred.test.ids, shape = (?, config.max_length_formula)

        """
        # scalar:embedding的维度
        embedding_dim = self._config.attn_cell_config.get("embedding_dim") # dim=50
        # embedding: [vocab_size, embedding_size]
        embedding_matrix = tf.get_variable("embedding_matrix",
                                           initializer=embedding_initializer(),
                                           shape=[self._n_tok, embedding_dim],
                                           dtype=tf.float32)
        # 作者果然是为start_token单独申请了变量
        start_token = tf.get_variable("start_token",
                                      dtype=tf.float32,
                                      shape=[embedding_dim],
                                      initializer=embedding_initializer())
        batch_size = tf.shape(img)[0]

        # training
        with tf.variable_scope("attn_cell", reuse=False):
            # formula:[batch, target_sequence_length]
            # embedding_matrix: [vocab_size, embedding_size]
            # decoder_input_embeddings: [batch, target_sequence_length+1, embedding_size]
            decoder_input_embeddings = get_embeddings(formula, embedding_matrix, embedding_dim, start_token, batch_size)
            # img: [batch, height, weight, channel]
            attn_meca = AttentionMechanism(img, self._config.attn_cell_config["dim_e"])
            rnn_cell = LSTMCell(num_units=self._config.attn_cell_config["num_units"])
            attn_cell = AttentionCell(rnn_cell, attn_meca, dropout, self._config.attn_cell_config, self._n_tok)

            # decoder_input_embeddings: [batch, target_sequence_length+1, embedding_size]
            # train_outputs: [batch, target_sequence_length+1, hidden_size]
            # _: [hidden=[batch, hidden_size], cell=[batch, hidden_size]]
            train_outputs, _ = tf.nn.dynamic_rnn(cell=attn_cell,
                                                 inputs=decoder_input_embeddings,
                                                 initial_state=attn_cell.initial_state())

        # inference(decoding)
        with tf.variable_scope("attn_cell", reuse=True):  # 注意:reuse=True
            attn_meca = AttentionMechanism(img=img, dim_e=self._config.attn_cell_config["dim_e"], tiles=self._tiles)
            rnn_cell = LSTMCell(num_units=self._config.attn_cell_config["num_units"], reuse=True)
            attn_cell = AttentionCell(rnn_cell, attn_meca, dropout, self._config.attn_cell_config, self._n_tok)
            if self._config.decoding == "greedy":
                # embedding: [vocab_size, embedding_size]
                decoder_cell = GreedyDecoderCell(embedding_matrix, attn_cell, batch_size, start_token, self._id_end)
            elif self._config.decoding == "beam_search":
                decoder_cell = BeamSearchDecoderCell(embedding_matrix, attn_cell, batch_size,
                        start_token, self._id_end, self._config.beam_size,
                        self._config.div_gamma, self._config.div_prob)

            test_outputs, _ = dynamic_decode(decoder_cell,
                    self._config.max_length_formula+1)

        return train_outputs, test_outputs


def get_embeddings(formula, embedding, embedding_size, start_token, batch_size):
    """Returns the embedding of the n-1 first elements in the formula concat
    with the start token

    Args:
        formula: (tf.placeholder) tf.uint32
        embedding: tf.Variable (matrix)
        embedding_size: (int) dimension of embeddings
        start_token: tf.Variable
        batch_size: tf variable extracted from placeholder

    Returns:
        embeddings_train: tensor

    """
    # embedding:[vocab_size, embedding_size] formula:[batch, target_sequence_length]
    # formula_:[batch, target_squence_length, embedding_size]
    formula_ = tf.nn.embedding_lookup(embedding, formula)
    # start_token:[1,1, embedding_size]
    start_token_ = tf.reshape(start_token, shape=[1, 1, embedding_size])
    # start_tokens: [batch, 1, embedding_size]
    start_tokens = tf.tile(start_token_, multiples=[batch_size, 1, 1])
    # formula:需要逆序,这样效果更好
    # embeddings: [batch, target_sequence_length+1, embedding_size]
    # TODO:感觉这里的start_token没有经过embeding矩阵,是不是有些问题
    # 好像又没有问题,作者单独为 start_token申请了矩阵
    embeddings = tf.concat([start_tokens, formula_[:, :-1, :]], axis=1)

    return embeddings


def embedding_initializer():
    """Returns initializer for embeddings"""
    def _initializer(shape, dtype, partition_info=None):
        E = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        E = tf.nn.l2_normalize(E, axis=-1)
        return E

    return _initializer
