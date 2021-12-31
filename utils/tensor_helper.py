import tensorflow as tf


# Padding方式填充矩阵
# base_size 基矩阵大小
# padding_size 填充后矩阵大小
def tensor_padding_layer(x, base_size, padding_size, channel_size, batch, name="padding-tensor"):
    ext_size = int((padding_size - base_size) / 2)
    padding_top_bottom = tf.constant(0.0, shape=[batch, ext_size, base_size, channel_size], dtype=tf.float32)
    padding_right_left = tf.constant(0.0, shape=[batch, padding_size, ext_size, channel_size], dtype=tf.float32)
    padding_input_x = tf.concat([padding_top_bottom, x, padding_top_bottom], axis=1, name=name)
    padding_input_x = tf.concat([padding_right_left, padding_input_x, padding_right_left], axis=2, name=name)
    return padding_input_x


def padding_tensor_by_size(x, size, in_channel, batch_size, name="padding-tensor"):
    ext_size = int(size / 2 - 1) if size % 2 == 0 else int((size - 1) / 2)
    padding_size = int(2 * ext_size + size)
    return tensor_padding_layer(x, size, padding_size, in_channel, batch_size, name), padding_size


def multiply_plus_b(x, w, b):
    return tf.multiply(x, w) + b
