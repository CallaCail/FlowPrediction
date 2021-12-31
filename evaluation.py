import tensorflow as tf
from data_helper import *
import numpy as np


class Evaluation:

    def __init__(self, folder, size):
        self.folder = folder
        self.steps = size

    def eval_model(self, t, r):
        checkpoint_file = tf.train.latest_checkpoint("model/{}/".format(self.folder))
        graph = tf.Graph()
        with graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                # 加载 .meta 图与变量
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.session, checkpoint_file)
                self.input_x = graph.get_operation_by_name("input-x").outputs[0]
                self.prediction = graph.get_operation_by_name("fc-layer/prediction").outputs[0]

                self.eval_data(self.steps, t, r)

    def eval_data(self, steps, t, r):

        predictions = []
        target = []

        for i in range(steps):
            data, label = eval_data(t, r)
            predicted = self.session.run([self.prediction], feed_dict={self.input_x: data})

            predictions.append(predicted)
            target.append(label)

            if (i % 100 == 0):
                print("{} steps have run.".format(i))

        MSE = np.mean(np.power((np.array(predictions) - np.array(target)), 2))
        RMSE = np.sqrt(MSE)
        print("MSE is {:4f}".format(MSE))
        print("RMSE is {:4f}".format(RMSE))

    def eval_map_data(self, steps, size, t, r):

        predictions = []
        target = []

        for i in range(steps):
            data, label = eval_map_data(t, r)
            predicted = self.session.run([self.prediction], feed_dict={self.input_x: data})

            predictions.append(predicted)
            target.append(label)

            if (i % 100 == 0):
                print("{} steps have run.".format(i))

        MSE = np.mean(np.power((np.array(predictions).reshape([steps, size * size]) - np.array(target).reshape([steps, size * size])), 2))
        RMSE = np.sqrt(MSE)
        print("MSE is {:4f}".format(MSE))
        print("RMSE is {:4f}".format(RMSE))


if __name__ == "__main__":
    evaluation = Evaluation("1555552018", size=1000)
    evaluation.eval_model(t=5, r=13)
