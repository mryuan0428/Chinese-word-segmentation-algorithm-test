# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math


# 隐藏层，或者叫全连接层，即计算 activation(Wx+b) 的一层
class HiddenLayer(object):
    """
    Hidden layer with or without bias.
    Input: tensor of dimension (dims*, input_dim)
    Output: tensor of dimension (dims*, output_dim)
    """

    def __init__(self, input_dim, output_dim, bias=True, activation='tanh', name='hidden_layer'):
        """
        :param input_dim: rnn_dim * 2（双向 RNN）
        :param output_dim: 标签种类（BEIS*POS）
        :param bias:
        :param activation:
        :param name:
        :return:
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_bias = bias
        self.name = name
        if activation == 'linear':
            self.activation = None
        elif activation == 'tanh':
            self.activation = tf.tanh
        elif activation == 'sigmoid':
            self.activation = tf.sigmoid
        elif activation == 'softmax':
            self.activation = tf.nn.softmax
        elif activation is not None:
            raise Exception('Unknown activation function: ' % activation)

        # Initialise weights and bias
        rand_uniform_init = tf.contrib.layers.xavier_initializer()
        self.weights = tf.get_variable(name + '_weights', [input_dim, output_dim], initializer=rand_uniform_init)
        self.bias = tf.get_variable(name + '_bias', [output_dim], initializer=tf.constant_initializer(0.0))

        # define parameters
        if self.is_bias:
            self.params = [self.weights, self.bias]
        else:
            self.params = [self.weights]

    def __call__(self, input_t):
        """
        :param input_t:
        :return:
        """
        self.input = input_t
        self.linear = tf.matmul(self.input, self.weights)
        if self.is_bias:
            self.linear += self.bias
        if self.activation is None:
            self.output = self.linear
        else:
            self.output = self.activation(self.linear)
        return self.output


class EmbeddingLayer(object):
    """
    Embedding layer to map input into word representations
    Input: tensor of dimension (dim*) with values in range(0, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """

    def __init__(self, input_dim, output_dim, weights=None, is_variable=False, trainable=True, name='embedding_layer'):
        """
        :param input_dim: 字表大小
        :param output_dim: 字向量维度，默认是64
        :param name:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.trainable = trainable
        self.weights = weights

        # Generate random embeddings or read pre-trained embeddings
        rand_uniform_init = tf.contrib.layers.xavier_initializer()
        if self.weights is None:
            self.embeddings = tf.get_variable(self.name + '_emb', [self.input_dim, self.output_dim],
                                              initializer=rand_uniform_init, trainable=self.trainable)
        elif is_variable:
            self.embeddings = weights
        else:
            emb_count = len(weights)
            if emb_count < input_dim:
                padd_weights = np.zeros([self.input_dim - emb_count, self.output_dim], dtype='float32')
                self.weights = np.concatenate((self.weights, padd_weights), axis=0)
            self.embeddings = tf.get_variable(self.name + '_emb', initializer=self.weights, trainable=self.trainable)
        # Define Parameters
        self.params = [self.embeddings]
        self.weight_name = self.name + '_emb'

    # 获取指定位置的向量，即 look up 操作
    def __call__(self, input_t):
        """
        return the embeddings of the given indexes
        :param input:
        :return:
        """
        self.input = input_t
        # tf.gather: Gather slices from params axis axis according to indices
        self.output = tf.gather(self.embeddings, self.input)
        return self.output


class Convolution(object):
    '''
    Regular convolutional layer
    '''

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)

    def __init__(self, conv_width, in_channels, out_channels, stride=1, dim=2, padding='SAME',
                 name='convolutional_layer'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        if dim == 1:
            self.strides = [1, stride, 1]
        else:
            self.strides = [1, stride, stride, 1]
        self.padding = padding
        self.name = name
        self.conv_width = conv_width
        if dim == 1:
            self.w_conv = self.weight_variable([self.conv_width, self.in_channels, self.out_channels],
                                               name=self.name + '_w')
        else:
            self.w_conv = self.weight_variable([self.conv_width, self.conv_width, self.in_channels, self.out_channels],
                                               name=self.name + '_w')
        self.b_conv = self.bias_variable([self.out_channels], name=self.name + '_b')

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=self.strides, padding=self.padding)

    def conv1d(self, x, W):
        return tf.nn.conv1d(x, W, stride=self.strides, padding=self.padding)

    def __call__(self, input_t):
        if self.dim == 1:
            return tf.nn.relu(self.conv1d(input_t, self.w_conv) + self.b_conv)
        else:
            return tf.nn.relu(self.conv2d(input_t, self.w_conv) + self.b_conv)


class Maxpooling(object):
    '''
    Maxpooling layer
    '''

    def __init__(self, pooling_size, stride=1, padding='SAME', name='pooling_layer'):
        self.padding = padding
        self.name = name
        self.ksize = [1, pooling_size, pooling_size, 1]

    def __call__(self, input_v):
        return tf.nn.max_pool(input_v, ksize=self.ksize, strides=self.ksize, padding='SAME')


class DropoutLayer(object):
    """
    Dropout layer
    """

    def __init__(self, p=0.5, name='dropout_layer'):
        """
        :param p: dropout rate
        :param name:
        """
        # assert 0. <= p < 1
        self.p = p
        self.name = name

    def __call__(self, input_t):
        self.input = input_t
        return tf.nn.dropout(self.input, keep_prob=1 - self.p, name=self.name)


class BiRNN(object):
    """
    Bidirectional LSTM
    """

    def __init__(self, cell_dim, nums_layers=1, p=0.5,
                 concat_output=True, output_state=False, name='BiRNN',
                 gru=True, scope="BiRNN"):
        """

        :param cell_dim: RNN 隐藏层神经元个数，默认设置是 200
        :param nums_layers: RNN 层数（深度）
        :param p: dropout 概率
        :param name: 每个 bucket 有一个 BiRNN，名称为 'BiRNN' + str(bucket)
        :param scope: “BiRNN”
        """
        self.cell_dim = cell_dim
        self.nums_layers = nums_layers
        self.p = p
        self.concat_output = concat_output
        self.output_state = output_state
        self.name = name
        self.scope = scope
        self.gru = gru
        with tf.variable_scope(scope):
            if gru:
                print "BiRNN %s uses GUR" % name
                fw_rnn_cell = tf.nn.rnn_cell.GRUCell(cell_dim)
                bw_rnn_cell = tf.nn.rnn_cell.GRUCell(cell_dim)
            else:
                print "BiRNN %s uses LSTM" % name
                fw_rnn_cell = tf.nn.rnn_cell.LSTMCell(cell_dim, state_is_tuple=True)
                bw_rnn_cell = tf.nn.rnn_cell.LSTMCell(cell_dim, state_is_tuple=True)

            # self.lstm_cell_fw = tf.contrib.rnn.AttentionCellWrapper(self.lstm_cell_fw, 5)
            # self.lstm_cell_bw = tf.contrib.rnn.AttentionCellWrapper(self.lstm_cell_bw, 5)
            # if self.p > 0.:
            self.fw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(fw_rnn_cell, output_keep_prob=(1 - self.p))
            self.bw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(bw_rnn_cell, output_keep_prob=(1 - self.p))

            if nums_layers > 1:
                self.fw_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([self.fw_rnn_cell] * nums_layers, state_is_tuple=True)
                self.bw_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([self.bw_rnn_cell] * nums_layers, state_is_tuple=True)

    def __call__(self, input_t, input_ids):
        """

        :param input_t: emb_out，每个字对应的字向量
        :param input_ids: input_v，每个字的 one-hot 向量，即 IDs，shape=（句子个数，句子长度）
        :return: RNN 的 output，shape=(句子个数，句子长度，rnn_dim*2)
        """
        self.input = input_t
        self.input_ids = input_ids

        # 计算每个句子的长度
        self.length = tf.reduce_sum(tf.sign(self.input_ids), axis=1)
        self.length = tf.cast(self.length, dtype=tf.int32)
        # outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output
        # output_fw: shape=(batch_size, max_time, cell_fw.output_size)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            self.fw_rnn_cell,
            self.bw_rnn_cell,
            self.input,
            sequence_length=self.length,
            dtype=tf.float32,
            scope=self.scope
        )
        if self.concat_output:
            self.output = tf.concat(values=outputs, axis=2)
            self.state = tf.concat(values=output_states, axis=1)
        else:
            self.output = outputs
            self.state = output_states
        if self.output_state:
            return self.output, self.state
        else:
            return self.output


class TimeDistributed(object):
    """
    Time-distributed wrapper for layers
    """

    def __init__(self, layer, name='Time-distributed Wrapper'):
        self.layer = layer
        self.name = name

    def __call__(self, input_t, input_ids=None, pad=None):
        """

        :param input_t: rnn_out
        :param input_ids:
        :param pad:
        :return:
        """
        # 在时间序列维度上 unpack，对每个时刻 RNN 的输出应用 layer()，
        # 然后在时间序列维度上 pack 回去
        # tf.unstack: Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
        self.input = tf.unstack(input_t, axis=1)
        if input_ids is None:
            self.out = [self.layer(splits) for splits in self.input]
        else:
            self.out = []
            pad = self.layer(self.input[0]) * 0
            masks = tf.reduce_sum(input_ids, axis=0)
            length = len(self.input)
            for i in range(length):
                r = tf.cond(tf.greater(masks[i], 0), lambda: self.layer(input_t[i]), lambda: pad)
                self.out.append(r)
        self.out = tf.stack(self.out, axis=1)
        return self.out


class Forward(object):
    """
    forward algorithm for the CRF loss
    """

    def __init__(self, observations, transitions, nums_tags, length, batch_size, viterbi=True):
        self.observations = observations
        self.transitions = transitions
        self.viterbi = viterbi
        self.length = length
        self.batch_size = batch_size
        self.nums_tags = nums_tags
        self.nums_steps = observations.get_shape().as_list()[1]

    @staticmethod
    def log_sum_exp(x, axis=None):
        """
        Sum probabilities in the log-space
        :param x:
        :param axis:
        :return:
        """
        x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, axis=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis))

    def __call__(self):
        small = -1000
        class_pad = tf.stack(small * tf.ones([self.batch_size, self.nums_steps, 1]))
        self.observations = tf.concat(axis=2, values=[self.observations, class_pad])
        b_vec = tf.cast(tf.stack(([small] * self.nums_tags + [0]) * self.batch_size), tf.float32)
        b_vec = tf.reshape(b_vec, [self.batch_size, 1, -1])
        self.observations = tf.concat(axis=1, values=[b_vec, self.observations])
        self.transitions = tf.reshape(tf.tile(self.transitions, [self.batch_size, 1]),
                                      [self.batch_size, self.nums_tags + 1, self.nums_tags + 1])
        self.observations = tf.reshape(self.observations, [-1, self.nums_steps + 1, self.nums_tags + 1, 1])
        self.observations = tf.transpose(self.observations, [1, 0, 2, 3])
        previous = self.observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1, self.nums_steps + 1):
            previous = tf.reshape(previous, [-1, self.nums_tags + 1, 1])
            current = tf.reshape(self.observations[t, :, :, :], [-1, 1, self.nums_tags + 1])
            alpha_t = previous + current + self.transitions
            if self.viterbi:
                max_scores.append(tf.reduce_max(alpha_t, axis=1))
                max_scores_pre.append(tf.argmax(alpha_t, axis=1))
            alpha_t = tf.reshape(self.log_sum_exp(alpha_t, axis=1), [-1, self.nums_tags + 1, 1])
            alphas.append(alpha_t)
            previous = alpha_t
        alphas = tf.stack(alphas, axis=1)
        alphas = tf.reshape(alphas, [-1, self.nums_tags + 1, 1])
        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.nums_steps + 1) + self.length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, self.nums_tags + 1, 1])
        max_scores = tf.stack(max_scores, axis=1)
        max_scores_pre = tf.stack(max_scores_pre, axis=1)
        return tf.reduce_sum(self.log_sum_exp(last_alphas, axis=1)), max_scores, max_scores_pre
