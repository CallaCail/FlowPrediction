# coding=utf-8

import numpy as np


def generate_circle(size, external_dim, internal_dim, in_channel):
    circle_filter = np.ones(shape=[in_channel, size, size], dtype=np.float32).tolist()

    ext_start = int((size - external_dim) / 2)
    ext_end = int(size - ext_start)
    for i in range(size):
        for j in range(size):
            for k in range(in_channel):
                if i < ext_start or j < ext_start or i >= ext_end or j >= ext_end:
                    circle_filter[k][i][j] = 0.0

    in_start = int((size - internal_dim) / 2)
    in_end = int(size - in_start)
    for i in range(in_start, in_end):
        for j in range(in_start, in_end):
            for k in range(in_channel):
                circle_filter[k][i][j] = 0.0

    return np.array(circle_filter)


if __name__ == '__main__':
    size = 6
    ccn = generate_circle(size, 6, 4, 2)
    print(ccn)
