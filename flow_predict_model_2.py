# coding=utf-8


import tensorflow as tf
import math

from circle_convolution_filter import *
from utils.tensor_helper import *


class Model:

    def __init__(self, x, t, r, batch_size=1):
        """
        :param x: 输入数据大小
        :param t: 卷积数据中时间长度
        :param r: 序列数据中时间长度
        :param batch_size: 批量大小
        """

        # self.batch_size = batch_size  # 批量训练数据的尺寸，训练数据是 X * X * T * batch_size 的四维数组

        # self.Y = 1  # 训练target的尺寸

        self.X = x  # 训练数的尺寸，X * X 的二维数组
        self.T = t  # 时间维度，训练数据是 X * X * T 的三维数组
        self.R = r  # 第四维度的时间长度

        self.input_y = tf.placeholder(tf.float32, shape=[1], name="input-y")
        self.input_x = tf.placeholder(tf.float32, shape=[self.R, self.X, self.X, self.T], name="input-x")

        # padding输入数据后，取部分区域用于训练
        self.input_x, size = padding_tensor_by_size(self.input_x, x, t, r)
        self.input_x = self.input_x[:, 30:62, 1:33]

        self.ccn_out = self.circle_cnn(self.X)
        reshape_out = tf.reshape(self.ccn_out, [1, self.R, self.X * self.X])
        self.rnn_out = self.time_rnn_layer(reshape_out)
        flatten_out = tf.layers.flatten(self.rnn_out)
        self.predict_layer(flatten_out)

    ################################################################################################################
    ################################################################################################################

    def map_predict_layer(self, input):
        self.prediction = self.fc_map_layer(input, input_size=self.X * self.X, hidden_size=self.X * self.X)
        flatten_y = tf.reshape(self.input_y, [-1, self.X * self.X])
        self.loss = self.loss_map_layer(self.prediction, flatten_y)

    def loss_map_layer(self, predict, target):
        with tf.name_scope("loss-layer"):
            return tf.sqrt(tf.reduce_mean(tf.square(predict - target)))

    def fc_map_layer(self, x, input_size, hidden_size):
        with tf.name_scope("fc-layer"):
            fc_input_w = tf.Variable(tf.truncated_normal([input_size, hidden_size], stddev=0.1))
            fc_input_b = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
            fc_input_out = tf.nn.xw_plus_b(x, fc_input_w, fc_input_b)

            fc_output_w = tf.Variable(tf.truncated_normal([hidden_size, input_size], stddev=0.1))
            fc_output_b = tf.Variable(tf.constant(0.1, shape=[input_size]))
            fc_output_out = tf.nn.xw_plus_b(fc_input_out, fc_output_w, fc_output_b)

            prediction = tf.reshape(fc_output_out, shape=[-1, input_size], name="prediction")

            return prediction

    ################################################################################################################
    ################################################################################################################

    def predict_layer(self, input):
        self.prediction = self.fc_layer(input, input_size=self.X * self.X, hidden_size=self.X)
        self.loss = self.loss_layer(self.prediction, self.input_y)

    def loss_layer(self, predict, target):
        with tf.name_scope("loss-layer"):
            return tf.sqrt(tf.reduce_mean(tf.square(predict - target)))

    def fc_layer(self, x, input_size, hidden_size):
        with tf.name_scope("fc-layer"):
            fc_input_w = tf.Variable(tf.truncated_normal([input_size, hidden_size], stddev=0.1))
            fc_input_b = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
            fc_input_out = tf.nn.xw_plus_b(x, fc_input_w, fc_input_b)

            fc_output_w = tf.Variable(tf.truncated_normal([hidden_size, 1], stddev=0.1))
            fc_output_b = tf.Variable(tf.constant(0.1, shape=[1]))
            fc_output_out = tf.nn.xw_plus_b(fc_input_out, fc_output_w, fc_output_b)

            prediction = tf.reshape(fc_output_out, shape=[-1, 1], name="prediction")

            return prediction

    def time_rnn_layer(self, rnn_input):
        with tf.name_scope("rnn-layer"):
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.X * self.X, name="time-rnn-layer-cell")
            init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            _, final_out = tf.nn.dynamic_rnn(rnn_cell, rnn_input, initial_state=init_state)
            return tf.nn.relu(final_out)

    ################################################################################################################
    ################################################################################################################
    # 环卷积序列层
    def circle_cnn(self, input_size):
        self.circle_out, circle_length = self.circle_convolution_layer(input_size)
        rnn_input = tf.reshape(self.circle_out, [self.R, circle_length, int(input_size * input_size)])
        rnn_out = self.circle_rnn_layer(rnn_input)
        return rnn_out

    # 环卷积层
    def circle_convolution_layer(self, size):
        length, multi_circle_out = self.multi_circle_convolution_layer(size)
        return multi_circle_out, length

    # 根据输入数据的大小，自动生成对应的环卷积核
    # 目前环卷积核的大小必须为正方形，且必须为大于等于3的奇数
    # 比如，5x5的输入数据，会生成5x5和3x3的环卷积核
    def multi_circle_convolution_layer(self, size):
        """
        :param
        size: 输入数据的尺寸
        :return: [环的数量, 环的序列]
        """

        start = 1
        if size % 2 == 0:
            start = 0

        cl_out_seq = []
        for i, index in enumerate(range(start, size, 2)):
            cl_out = self.circle_convolution_filter(index + 2, index, self.T, name="ccn_" + str(i))
            cl_out_seq.insert(0, cl_out)
        return int(size / 2), tf.stack(cl_out_seq, axis=1)

    # 环卷积核
    # external_dim 环的外径
    # internal_dim 环的内径
    # in_channel 输入通道的尺寸
    # out_channel 输出通道的尺寸
    # 使用ReLU和drop保证只有环上的权重可以被更新
    def circle_convolution_filter(self, external_dim, internal_dim, in_channel, name="circle_filter"):
        circle_filter = generate_circle(self.X, external_dim, internal_dim, in_channel).swapaxes(0, 2).swapaxes(0, 1)
        cl_drop = tf.constant(value=circle_filter, dtype=tf.float32, name=name + '_drop')
        cl_weight = tf.Variable(initial_value=circle_filter, dtype=tf.float32, name=name + '_weight')
        cl_bias = tf.Variable(tf.constant(1., shape=[1], dtype=tf.float32), name=name + '_bias')
        cl_out = tf.nn.relu((cl_weight * self.input_x + cl_bias) * cl_drop, name=name + '_out')
        cl_transpose = tf.transpose(cl_out, perm=[0, 3, 1, 2])
        cl_reduce = tf.reduce_sum(cl_transpose, axis=1)
        return tf.reshape(cl_reduce, [self.R, self.X * self.X])

    # 环序列层
    # rnn_input 环卷积核层提取的特征
    # 环卷积产生的数据放入RNN中，由序列网络提取新的特征
    def circle_rnn_layer(self, rnn_input):
        with tf.name_scope("circle-rnn-layer"):
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.X * self.X, name="circle-rnn-layer-cell")
            init_state = rnn_cell.zero_state(batch_size=self.R, dtype=tf.float32)
            _, rnn_out = tf.nn.dynamic_rnn(rnn_cell, rnn_input, initial_state=init_state)
            return tf.nn.tanh(rnn_out)

    ################################################################################################################
    ################################################################################################################
    # def circle_layer(self):
    #     self.circle_conv_filter = tf.Variable(initial_value=size_3())
    #     strides = [1, 1, 1, 1]
    #     cw = tf.nn.conv2d(input=self.input_x, filter=self.circle_conv_filter, strides=strides, padding="SAME",
    #                       name="cw")
    #     cb = tf.Variable(tf.constant(0.1, shape=[1]), name="cb")
    #     self.circle_layer_out = tf.nn.relu(tf.nn.bias_add(cw, cb))
    #
    # def circle_convolution_layer_test(self):
    #     self.cl_drop = tf.constant(value=size_1(), dtype=tf.float32)
    #     self.cl_weight = tf.Variable(initial_value=size_1(), dtype=tf.float32, name="cw")
    #     self.cl_bias = tf.Variable(tf.constant(1., tf.float32, shape=[1]), name="cb")
    #
    #     circle_tmp_out = self.cl_weight * self.input_x + self.cl_bias
    #
    #     self.cl_out = tf.nn.relu(circle_tmp_out * self.cl_drop)
    #
    #     self.loss = tf.reduce_mean(tf.square(self.cl_out - self.input_y))
    ################################################################################################################
    ################################################################################################################
