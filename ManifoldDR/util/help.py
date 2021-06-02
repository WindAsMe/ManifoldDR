import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from ManifoldDR.DE.MyProblem import Block_Problem
import geatpy as ea
import os.path as path


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


