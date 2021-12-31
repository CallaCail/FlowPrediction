import random

import h5py
import numpy as np

def generate_data(t, r):
    f = h5py.File('data/BJ13_M32x32_T30_InOut.h5')
    data = []
    label = []
    random_time = random.randint(1, 3000 - max(t, r))
    # print('选取随机时间段：' + str(random_time))
    for i in range(0, r):
        arr = []
        for j in range(0, t):
            arr.append(f['data'][random_time + (t - j) + (r - i)][0])
        arr1 = np.swapaxes(arr, axis1=0, axis2=2)
        arr2 = np.swapaxes(arr1, axis1=0, axis2=1)
        data.append(arr2)
    # 后两个维度表示矩阵平面中的坐标
    label.append(f['data'][random_time + j + i + 1][0][31][1])
    # print(np.shape(data))
    # print(np.shape(label))
    return data, label


def eval_data(t, r):
    f = h5py.File('data/BJ13_M32x32_T30_InOut.h5')

    data = []
    label = []

    random_time = random.randint(3000, 4880 - max(t, r))

    for i in range(0, r):
        arr = []
        for j in range(0, t):
            arr.append(f['data'][random_time + (t - j) + (r - i)][0])
        arr1 = np.swapaxes(arr, axis1=0, axis2=2)
        arr2 = np.swapaxes(arr1, axis1=0, axis2=1)
        data.append(arr2)
    label.append(f['data'][random_time + j + i + 1][0][15][15])

    return data, label


def generate_map_data(t, r):
    f = h5py.File('data/BJ13_M32x32_T30_InOut.h5')
    data = []
    # label = []
    random_time = random.randint(1, 3000 - max(t, r))
    # print('选取随机时间段：' + str(random_time))
    for i in range(0, r):
        arr = []
        for j in range(0, t):
            arr.append(f['data'][random_time + (t - j) + (r - i)][0])
        arr1 = np.swapaxes(arr, axis1=0, axis2=2)
        arr2 = np.swapaxes(arr1, axis1=0, axis2=1)
        data.append(arr2)
    label = f['data'][random_time + j + i + 1][0]
    # print(np.shape(data))
    # print(np.shape(label))
    return data, label


def eval_map_data(t, r):
    f = h5py.File('data/BJ13_M32x32_T30_InOut.h5')
    data = []
    # label = []
    random_time = random.randint(3000, 4880 - max(t, r))
    # print('选取随机时间段：' + str(random_time))
    for i in range(0, r):
        arr = []
        for j in range(0, t):
            arr.append(f['data'][random_time + (t - j) + (r - i)][0])
        arr1 = np.swapaxes(arr, axis1=0, axis2=2)
        arr2 = np.swapaxes(arr1, axis1=0, axis2=1)
        data.append(arr2)
    label = f['data'][random_time + j + i + 1][0]
    # print(np.shape(data))
    # print(np.shape(label))
    return data, label


if __name__ == "__main__":
    # print(generate_data(8, 5))
    print(eval_data(5, 13))
