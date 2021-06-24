from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import geatpy as ea
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from ManifoldDR.DE import MyProblem
from pykrige.ok import OrdinaryKriging
from ManifoldDR.util import help
import copy
import heapq
import time


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


if __name__ == '__main__':

    data = np.random.randint(-50, 50, (500, 10))
    DR = umap.UMAP(n_components=2)
    methods = ['UMAP']
    low_D_data = DR.fit_transform(data)
    range_step = 0.01
    n_best = 10
    k = 10

    Functions = [Ackley, Schwefel, Rastrigin]
    for f in Functions:
        fitness = []
        for d in data:
            fitness.append(f(d))
        print('Random fitness: ', np.average(fitness), min(fitness))
        best_index = np.argmin(fitness)

        good_point_origin = np.append(low_D_data[best_index], fitness[best_index])
        draw_3D_Point(low_D_data, fitness, good_point_origin)
        x_range = [min(low_D_data[:, 0]), max(low_D_data[:, 0])]
        y_range = [min(low_D_data[:, 1]), max(low_D_data[:, 1])]
        gridx = np.arange(x_range[0], x_range[1], range_step)
        gridy = np.arange(y_range[0], y_range[1], range_step)


        # Create Krige model and find best n coordinate in model
        # This process is instead of model optimization before
        k3d1 = Krige_model(gridx, gridy, low_D_data, fitness)
        indexes = find_n_matrix(k3d1, n_best, gridx, gridy)

        # draw_3D_Point_Best(low_D_data, fitness, good_point_origin, np.array(indexes), best_fitness)
        time1 = time.time()
        high_D_best_pop_real = help.inverse_simulate(data, low_D_data, indexes, k)
        time2 = time.time()
        fit = []
        for chrom in high_D_best_pop_real:
            fit.append(f(chrom))
        print('Simulate inverse: ', np.average(fit), min(fit), 'time: ', time2-time1)
        time3 = time.time()
        high_D_best_pop_inverse = DR.inverse_transform(indexes)
        time4 = time.time()
        fit = []
        for chrom in high_D_best_pop_inverse:
            fit.append(f(chrom))
        print('UMAP inverse: ', np.average(fit), min(fit), 'time: ', time4-time3)



