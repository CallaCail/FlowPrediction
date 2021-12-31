# coding=utf-8

import tensorflow as tf
import numpy as np
from flow_predict_model_2 import Model
from data_helper import *
import time

# data = np.array([[[[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
#                   [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
#                   [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
#                   [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
#                   [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]],
#                  [[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]],
#                   [[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]],
#                   [[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]],
#                   [[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]],
#                   [[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]], dtype=np.float32)
#
# target = np.array([1.], dtype=np.float32)

size = 32
t = 5
r = 13

model = Model(size, t, r, batch_size=1)

global_step = tf.Variable(0, name="global_step", trainable=False)
train_step = tf.train.AdamOptimizer().minimize(model.loss, global_step=global_step)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1000):
        data, label = generate_data(t, r)

        _, loss, circle_out = session.run(
            [train_step, model.loss, model.circle_out],
            feed_dict={model.input_x: data, model.input_y: label}
        )

        if i % 100 == 0:
            print("{} steps have run. loss is {:4f}".format(i, loss))

    out = np.array(circle_out).reshape([-1, 16, 32, 32])
    for line in out[0][0]:
        for item in line:
            print(item, end='\t')
        print("\n")

    print(out.shape)
    print(np.array(circle_out).shape)

    current_step = tf.train.global_step(session, global_step)
    timestamp = str(int(time.time()))
    saver.save(session, "model/" + timestamp + "/model", global_step=current_step)
