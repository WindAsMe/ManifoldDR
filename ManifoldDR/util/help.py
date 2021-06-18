import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from ManifoldDR.DE.MyProblem import Block_Problem
from ManifoldDR.DE import MyProblem
import geatpy as ea
import os.path as path
import heapq
import umap
from sklearn.decomposition import PCA, NMF
import time
from pykrige.ok import OrdinaryKriging


def create_local_model_data(scale, total_dim, group_dim, current_index, scale_range):
    total_data = np.zeros((scale, total_dim))
    for d in total_data:
        for i in range(current_index*group_dim, (current_index+1)*group_dim):
            d[i] = random.uniform(scale_range[0], scale_range[1])
    real_data = total_data[:, current_index*group_dim:(current_index+1)*group_dim]
    return total_data, real_data


def create_data(scale, dim, scale_range):
    data = []
    for j in range(0, scale):
        temp = []
        for i in range(0, dim):
            temp.append(random.uniform(scale_range[0], scale_range[1]))
        data.append(temp)
    return np.array(data, dtype='double')


def create_result(train_data, f):
    result = []
    for x in train_data:
        result.append(f(x))
    return np.array(result)


# Change the features ['1', '1^2', '1 2^2'] to [[1], [1, 1], [1, 2, 2]]
def feature_names_normalization(feature_names):
    result = []
    for s in feature_names:
        if s.isdigit():
            result.append([int(s)])
            continue
        else:
            temp = []
            s_list = s.split(' ')
            for sub_s in s_list:
                if sub_s.isdigit():
                    temp.append(int(sub_s))
                else:
                    sub_s_list = sub_s.split('^')
                    for i in range(int(sub_s_list[1])):
                        temp.append(int(sub_s_list[0]))
            result.append(temp)
    return result


def not_zero_feature(coef, feature_names):
    new_coef = []
    new_feature_names = []
    for i in range(len(coef)):
        if coef[i] != 0:
            new_coef.append(coef[i])
            new_feature_names.append(feature_names[i])
    return new_coef, new_feature_names


def have_same_element(l1, l2):
    for e in l1:
        if e in l2:
            return True
    return False


def list_combination(l1, l2):
    for l in l2:
        if l not in l1:
            l1.append(l)
    return l1


def group_DFS(Dim, feature_names, max_variable_num):
    temp_feature_names = copy.deepcopy(feature_names)
    groups_element = []
    groups_index = []

    while temp_feature_names:
        elements = temp_feature_names.pop(0)
        group_element = elements
        group_index = [feature_names.index(elements)]
        flag = [1]
        for element in elements:
            help_DFS(group_element, group_index, element, temp_feature_names, feature_names, flag, max_variable_num)
        group_element = list(set(group_element))
        interactions = []
        for name in temp_feature_names:
            interaction = [a for a in group_element if a in name]
            if len(interaction) > 0:
                interactions.append(name)
        for name in interactions:
            temp_feature_names.remove(name)
        groups_element.append(group_element)
        groups_index.append(group_index)

    verify = []
    for group in groups_element:
        verify.extend(group)
    final_g = []
    for i in range(Dim):
        if i not in verify:
            final_g.append(i)
    groups_element.append(final_g)
    return groups_element


def help_DFS(group_element, group_index, element, temp_feature_names, feature_names, flag, max_variable_num):
    if flag[0] >= max_variable_num:
        return
    else:
        i = -1
        while temp_feature_names:
            i += 1
            if i >= len(temp_feature_names):
                return
            else:
                if element in temp_feature_names[i]:
                    temp_elements = temp_feature_names.pop(i)
                    group_element.extend(temp_elements)
                    group_index.append(feature_names.index(temp_elements))

                    flag[0] = len(set(group_element))
                    if flag[0] >= max_variable_num:
                        return
                    for temp_element in temp_elements:
                        help_DFS(group_element, group_index, temp_element, temp_feature_names, feature_names, flag,
                                 max_variable_num)
                        if flag[0] >= max_variable_num:
                            return


def write_obj_trace(p, fileName, trace):
    this_path = path.realpath(__file__)
    data_path = path.dirname(path.dirname(this_path)) + '\\data\\trace\\obj\\' + p + '\\' + fileName

    with open(data_path, 'a') as f:
        f.write('[')
        for i in range(len(trace)):
            if i == len(trace) - 1:
                f.write(str(trace[i]))
            else:
                f.write(str(trace[i]) + ', ')
        f.write('],')
        f.write('\n')
        f.close()


def write_info(p, fileName, data):
    this_path = path.realpath(__file__)
    data_path = path.dirname(path.dirname(this_path)) + '\\data\\trace\\obj\\' + p + '\\' + fileName
    with open(data_path, 'a') as f:
        f.write(data + ', ')
        f.write('\n')
        f.close()


def write_CPU_cost(p, fileName, data):
    this_path = path.realpath(__file__)
    data_path = path.dirname(path.dirname(this_path)) + '\\data\\trace\\obj\\' + p + '\\' + fileName
    with open(data_path, 'a') as f:
        f.write(data + ', ')
        f.write('\n')
        f.close()


def write_EFS_cost(p, fileName, data):
    this_path = path.realpath(__file__)
    data_path = path.dirname(path.dirname(this_path)) + '\\data\\trace\\obj\\' + p + '\\' + fileName
    with open(data_path, 'a') as f:
        f.write(data + ', ')
        f.write('\n')
        f.close()


def preserve(var_traces, benchmark_function):
    obj_traces = []
    for v in var_traces:
        obj_traces.append(benchmark_function(v))
    for i in range(len(obj_traces) - 1):
        if obj_traces[i] < obj_traces[i + 1]:
            var_traces[i + 1] = var_traces[i]
            obj_traces[i + 1] = obj_traces[i]
    return var_traces, obj_traces


def draw_summary(x, x_DECC_D, x_DECC_DG, x_DECC_L, x_DECC_CL, Normal_ave, One_ave, Random_ave, DECC_D_ave,
                 DECC_DG_ave, LASSO_ave, DECC_CL_ave):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.semilogy(x, Normal_ave, label='従来法1', linestyle=':')
    plt.semilogy(x, One_ave, label='従来法2', linestyle=':', color='plum')
    plt.semilogy(x, Random_ave, label='従来法3', linestyle=':', color='aqua')
    plt.semilogy(x_DECC_D, DECC_D_ave, label='従来法4', linestyle=':', color='grey')
    plt.semilogy(x_DECC_DG, DECC_DG_ave, label='従来法5', linestyle=':', color='lawngreen')
    plt.semilogy(x_DECC_L, LASSO_ave, label='提案法1', color='red')
    plt.semilogy(x_DECC_CL, DECC_CL_ave, label='提案法2', color='orange')

    # plt.plot(x, Normal_ave, label='従来法1', linestyle=':')
    # plt.plot(x, One_ave, label='従来法2', linestyle=':', color='aqua')
    # plt.plot(x, Random_ave, label='従来法3', linestyle=':')
    # plt.plot(x_DECC_D, DECC_D_ave, label='従来法4', linestyle=':', color='grey')
    # plt.plot(x_DECC_DG, DECC_DG_ave, label='従来法5', linestyle=':', color='lawngreen')
    # plt.plot(x_DECC_L, LASSO_ave, label='提案法1', color='red')
    # plt.plot(x_DECC_CL, DECC_CL_ave, label='提案法2', color='orange')
    font_title = {'size': 18}
    font = {'size': 16}
    plt.title('$f_{15}$', font_title)
    plt.xlabel('Fitness evaluation times (×${10^6}$)', font)
    plt.ylabel('Fitness', font)
    plt.legend()
    # plt.savefig(
    #    'D:\CS2019KYUTAI\PythonProject\SparseModeling\data\\pic\\' + name + '_obj')

    plt.text(3, 0.3*10e10, "**", fontdict={'size': 14, 'color': 'red'})
    plt.text(3, 0.2*10e10, "**", fontdict={'size': 14, 'color': 'orange'})

    plt.show()


# Return: True(separable) False(none-separable)
# e1 < e2
def Differential(e1, e2, function, intercept, one_bias):
    index = np.zeros((1, 1000))[0]
    index[e1] = 1
    index[e2] = 1

    c = function(index) - intercept
    b = one_bias[e1]
    a = one_bias[e2]
    return np.abs(c - (a + b)) < 0.001


def DG_Differential(e1, e2, a, function, intercept):
    index1 = np.zeros((1, 1000))[0]
    index2 = np.zeros((1, 1000))[0]
    index1[e2] = 1
    index2[e1] = 1
    index2[e2] = 1

    b = function(index1) - intercept
    c = function(index2) - intercept

    return np.abs(c - (a + b)) < 0.001


# Return: True(separable) False(none-separable)
# e1 < e2
def LIDI_R(e1, e2, function, intercept, one_bias):
    index = np.zeros((1, 1000))[0]
    index[e1] = 1
    index[e2] = 1
    # intercept: f(0,0)
    c = function(index)-intercept  # f(1,1)-f(0,0)
    b = one_bias[e1]  # f(0,1)-f(0,0)
    a = one_bias[e2]  # f(1,0)-f(0,0)

    return signal(c-b) == signal(a) and signal(c-a) == signal(b)


def signal(delta):
    if delta > 0:
        return 1
    elif delta == 0:
        return 0
    else:
        return -1


# Return True means proper
def check_proper(groups):
    flag = [False] * 1000
    for group in groups:
        for e in group:
            flag[e] = True
    return False not in flag


# False: stop
def is_Continue(Generations, threshold=0.001):
    for i in range(0, len(Generations)-1):
        if Generations[i] < Generations[i+1]:
            Generations[i+1] = Generations[i]
    flag = [True] * (len(Generations) - 1)
    for i in range(len(Generations) - 1):
        if Generations[i + 1] * (1 + threshold) > Generations[i]:
            flag[i] = False
    return True in flag


def initial_population(NIND, groups, up, down, elite=None):
    initial_Population = []
    for group in groups:
        problem = Block_Problem(group, None, up, down, None)  # 实例化问题对象

        Encoding = 'RI'  # 编码方式
        NIND = NIND      # 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        population = ea.Population(Encoding, Field, NIND)
        population.initChrom(NIND * len(group))
        if elite is not None:
            for i in range(len(population.Chrom[0])):
                population.Chrom[0][i] = elite[group[i]]
        initial_Population.append(population)
    return initial_Population


def draw_check(x, data, name):
    plt.plot(x, data, label=name)
    plt.xlabel('Evaluation times')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()


def Normalization(m, iter):
    for j in range(len(m[0])):
        for i in range(iter):
            if m[i][j] == 0:
                m[i][j] = m[i-1][j]
    return m


def draw_error(f):
    up_error = []
    down_error = []
    for i in range(0, 10):
        up_error.append(f[i, 1] - f[i, 3])
        down_error.append(f[i, 3] - f[i, 2])

    font_title = {'size': 18}
    font = {'size': 16}
    plt.title('$f_{13}$', font_title)
    plt.xlabel('Iteration times', font)
    plt.ylabel('Percentage', font)
    plt.errorbar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], f[:, 3], yerr=[down_error, up_error], fmt="o")
    plt.show()


def filling(Dim, data, group):
    temp_data = np.zeros((len(data), Dim))
    data = np.array(data)
    for i in range(len(group)):
        temp_data[:, group[i]] = data[:, i]
    return temp_data


def find_n_best(Chroms, ObjVs, n):
    new_Chroms = []
    new_ObjVs = []
    index = ObjVs.argsort()
    for i in range(n):
        new_Chroms.append(Chroms[index[i]])
        new_ObjVs.append([ObjVs[index[i]]])
    return np.array(new_Chroms), np.array(new_ObjVs)


# To reduce the CPU time
def able_inverse_simulate(DR, high_D_origin_pop, low_D_origin_pop, low_D_best_pop, low_D_best_index, k):
    high_D_best_pop_sti = inverse_simulate(high_D_origin_pop, low_D_origin_pop, [low_D_best_pop[low_D_best_index]], k)
    high_D_best_pop_real = DR.inverse_transform(low_D_best_pop)
    high_D_best_pop_real[low_D_best_index] = high_D_best_pop_sti[0]

    return high_D_best_pop_real


# high_D_origin_pop: original population information in high D
# low_D_origin_pop: original population information in low D
# low_D_best_pop: best population information in low D
# k: k nearest parameter(must < len of matrix)

# Return: best population information in high D
def inverse_simulate(high_D_origin_pop, low_D_origin_pop, low_D_best_pop, k):
    up = []
    down = []
    high_D_origin_pop = np.array(high_D_origin_pop)
    for i in range(len(high_D_origin_pop[0])):
        up.append(max(high_D_origin_pop[:, i]))
        down.append(min(high_D_origin_pop[:, i]))

    low_D_k_Normal_dis, k_index = k_dis_index(low_D_origin_pop, low_D_best_pop, k)
    high_D_best_pop = []
    for i in range(len(low_D_best_pop)):
        high_D_best_pop.append(list(distance_Optimization(len(high_D_origin_pop[0]), 1000, loss, up, down, low_D_k_Normal_dis[i],
                                                     k_index[i], high_D_origin_pop)))
    return high_D_best_pop


# delta = ∑_(𝑖=1)^𝑛((𝑑 ̂_𝑖−𝑑_𝑖))^2
def loss(low_D_k_Normal_dis, k_index, high_D_origin_pop, individual):
    real_dis = []
    for index in k_index:
        real_dis.append(distance(high_D_origin_pop[index], individual))
    high_D_k_Normal_dis = regularization(real_dis)
    delta = 0
    for i in range(len(high_D_k_Normal_dis)):
        delta += (high_D_k_Normal_dis[i] - low_D_k_Normal_dis[i])**2

    return delta


# low_D_origin_pop: original population(Low D)
# low_D_best_pop: population when optimization stop(Low D)
# k: k nearest parameter(must < len of matrix)
# Return: normalized k-neighbor distance and index
def k_dis_index(low_D_origin_pop, low_D_best_pop, k):
    if k > len(low_D_origin_pop):
        return None
    # k_index: save the coordinate of k_neighbor
    # k_Normal_dis: save the normalized distance of k_neighbor
    k_index = []
    low_D_k_Normal_dis = []
    for point_elite in low_D_best_pop:
        point_distances = []
        for point_original in low_D_origin_pop:
            point_distances.append(distance(point_original, point_elite))
        dis, index = k_nearest(point_distances, k)
        low_D_k_Normal_dis.append(dis)
        k_index.append(index)
    return low_D_k_Normal_dis, k_index


def distance_Optimization(Dim, MAX_iteration, loss, up, down, low_D_k_Normal_dis, k_index, high_D_origin_pop):
    problem = MyProblem.distanceProblem(Dim, loss, up, down, low_D_k_Normal_dis, k_index, high_D_origin_pop)  # 实例化问题对象
    """===========================算法参数设置=========================="""
    Encoding = 'RI'  # 编码方式
    NIND = 50  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    population.initChrom()

    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])

    return var_trace[np.argmin(obj_trace[:, 1])]


# Find the k nearest points
def k_nearest(distances, k):
    k_dis = heapq.nsmallest(k, distances)
    k_index = []
    for dis in k_dis:
        k_index.append(distances.index(dis))
    k_Normalize_dis = regularization(k_dis)
    return k_Normalize_dis, k_index


def regularization(k_distances):

    k_Normalize_dis = []
    dis_sum = sum(k_distances)
    for dis in k_distances:
        k_Normalize_dis.append((dis/dis_sum)**2)
    return k_Normalize_dis


def distance(point1, point2):
    dis = 0
    for i in range(len(point1)):
        dis += (point1[i] - point2[i])**2
    return np.sqrt(dis)


def data_split(data, z, percentage):
    sorted_z = sorted(z)
    split_fitness = sorted_z[int(len(z) * percentage)]
    better_data = []
    better_z = []
    worse_data = []
    worse_z = []
    for index, fitness in enumerate(z):
        if fitness < split_fitness:
            better_data.append(data[index])
            better_z.append(fitness)
        else:
            worse_data.append(data[index])
            worse_z.append(fitness)
    return better_data, better_z, worse_data, worse_z


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
    best_fitness = []
    for i in best_index:
        row_x, column_y = matrix_index(i, len(gridy))
        if row_x > len(gridx) - 1 or column_y > len(gridy) - 1:
            print(i, row_x, column_y, len(gridx), len(gridy), matrix.shape)
        indexes.append([gridx[row_x], gridy[column_y]])
        best_fitness.append(temp_matrix[i])
    return indexes, best_fitness

