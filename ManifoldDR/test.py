from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import geatpy as ea
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from ManifoldDR.DE import MyProblem
from ManifoldDR.util import help
import copy


def Regression(degree, train_data, train_label):
    poly_reg = PolynomialFeatures(degree=degree)
    train_data_ploy = poly_reg.fit_transform(train_data)
    reg = LinearRegression()
    reg.fit(train_data_ploy, train_label)
    return reg


def function(data):
    z = []
    for i in range(len(data)):
        z.append(data[i][0]**2 + data[i][1]**2)
    return z, np.argmin(z)


def draw_3D(x, y, z):
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='aqua')
    plt.show()


def draw_3D_Point(x, y, z, point):
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='aqua')
    ax.scatter3D(point[0], point[1], point[2], color='black')
    plt.show()


def draw_2D(degree, x, y, point, reg, method, up, down, best_point):
    data = np.linspace(down, up, 500).reshape(-1, 1)
    fitness = reg.predict(PolynomialFeatures(degree=degree).fit_transform(data))
    plt.plot(data, fitness)
    plt.scatter(x, y, color='aqua', label=method)
    plt.scatter(point[0], point[1], color='black', label='Real Optimum')
    plt.scatter(best_point[0], best_point[1], color='red', label='Optimum in model')
    plt.legend()
    plt.show()


def model_Optimization(degree, MAX_iteration, model, up, down, data):
    problem = MyProblem.modelProblem(degree, len(data[0]), model, up, down)  # 实例化问题对象

    """===========================算法参数设置=========================="""
    Encoding = 'RI'  # 编码方式
    NIND = len(data)  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    population.initChrom()
    population.Chrom = data

    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])

    return var_trace, obj_trace[:, 1], population


def draw_3D_Restore(x, y, z, point, x_1, y_1, z_1, point_1):
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='aqua')
    ax.scatter3D(point[0], point[1], point[2], color='black')

    ax.scatter3D(x_1, y_1, z_1, color='navy')
    ax.scatter3D(point_1[0], point_1[1], point_1[2], color='red')
    plt.show()


def enborder(up, down, chrom):
    temp_chrom = copy.deepcopy(chrom)
    for j in range(len(temp_chrom[0])):
        for i in range(len(temp_chrom)):
            if temp_chrom[i][j] > up[j]:
                temp_chrom[i][j] = up[j]
            elif temp_chrom[i][j] < down[j]:
                temp_chrom[i][j] = down[j]
    return temp_chrom


if __name__ == '__main__':
    x = np.random.randint(-10, 10, (200, 1))
    y = np.random.randint(-10, 10, (200, 1))
    data = np.hstack((x, y))
    z, min_individual = function(data)
    point_high = [x[min_individual], y[min_individual], z[min_individual]]
    # draw_3D(x, y, z)
    draw_3D_Point(x, y, z, point_high)

    # TSNE(n_components=1)
    DRs = [umap.UMAP(n_components=1)]
    methods = ['UMAP']
    degree = 4
    k = 3
    for index, DR in enumerate(DRs):
        low_D_data = DR.fit_transform(data)
        reg = Regression(degree, low_D_data, z)
        point = [low_D_data[min_individual][0], z[min_individual]]
        up = [max(low_D_data[:, 0])]
        down = [min(low_D_data[:, 0])]
        temp_low_D_data = enborder(up, down, low_D_data)
        var_trace, obj_trace, population = model_Optimization(degree, 10, reg, up, down, temp_low_D_data)


        best_point = [var_trace[len(var_trace)-1][0], obj_trace[len(obj_trace)-1]]
        print(best_point)
        draw_2D(degree, low_D_data[:, 0], z, point, reg, methods[index], up, down, best_point)

        high_D_best_pop_real = np.array(help.inverse_simulate(data, low_D_data, population.Chrom, k))
        z_1, temp = function(high_D_best_pop_real)

        point_1 = [high_D_best_pop_real[temp][0], high_D_best_pop_real[temp][0], z_1[temp]]
        draw_3D_Restore(x, y, z, point_high, high_D_best_pop_real[:, 0], high_D_best_pop_real[:, 1], z_1, point_1)




