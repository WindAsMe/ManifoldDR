from matplotlib import pyplot as plt
import numpy as np
from pykrige.ok import OrdinaryKriging

# 10D
def Ackley(data):
    item1 = 0
    item2 = 0
    for d in data:
        item1 += d * d
        item2 += np.cos(2 * np.pi * d)
    item1 = -20 * np.exp(-0.2 * np.sqrt((1 / len(data)) * item1))
    item2 = np.exp((1 / len(data)) * item2)
    return item1 - item2 + 20 + np.e


# 10D
def Rastrigin(data):
    item = 0
    for d in data:
        item += d * d - 10 * np.cos(2 * np.pi * d)
    return item + 10 * len(data)


# 10D
def Schwefel(data):
    item = 0
    for d in data:
        item += d * np.sin(np.sqrt(np.abs(d)))
    return 418.9829 * len(data) - item


def draw_3D_Point(data, fitness, point):
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], fitness, color='aqua', label='Original points')
    ax.scatter3D(point[0], point[1], point[2], color='black', label='Good point')
    plt.legend()
    plt.show()


def draw_3D_Point_Best(data, fitness, point, individual, value):
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], fitness, color='aqua', label='Original points')
    ax.scatter3D(point[0], point[1], point[2], color='black', label='Good point')
    ax.scatter3D(individual[:, 0], individual[:, 1], value, color='red', label='Optimum point')
    plt.legend()
    plt.show()


# For 3D
def Krige_model(gridx, gridy, data, fitness):
    ok3d = OrdinaryKriging(data[:, 0], data[:, 1], fitness, variogram_model='hole-effect')  # 模型
    # pykrige提供 linear, power, gaussian, spherical, exponential, hole-effect几种variogram_model可供选择，默认的为linear模型。
    k3d1, ss3d = ok3d.execute("grid", gridx, gridy)
    return k3d1


def matrix_index(index, dim):
    row = int(index / dim)
    column = index % dim
    return row, column


# This function is to process the result of Krige
# Output is the indexes of best n
def find_n_matrix(matrix, n, gridx, gridy):
    temp_matrix = matrix.flatten()
    best_index = temp_matrix.argsort()[:n]
    indexes = []
    for i in best_index:
        row_x, column_y = matrix_index(i, len(gridy))
        indexes.append([gridx[row_x], gridy[column_y]])
    return indexes

