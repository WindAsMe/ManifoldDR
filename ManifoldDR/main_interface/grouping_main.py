from ManifoldDR.util import help
from ManifoldDR.model import SparseModel
from ManifoldDR.DE import DE
from cec2013lsgo.cec2013 import Benchmark
import random
import numpy as np


def LASSOCC(func_num):
    Dim = 1000
    group_dim = 50
    size = group_dim * 100
    degree = 3
    bench = Benchmark()
    max_variables_num = group_dim

    function = bench.get_function(func_num)
    benchmark_summary = bench.get_info(func_num)
    scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
    verify_time = 0
    All_groups = []

    intercept = function(np.zeros((1, 1000))[0])
    one_bias = []
    for i in range(1000):
        index = np.zeros((1, 1000))[0]
        index[i] = 1
        one_bias.append(function(index) - intercept)
    verify_time += 1001

    for current_index in range(0, int(Dim / group_dim)):
        # print(current_index)
        Lasso_model, Feature_names = SparseModel.Regression(degree, size, Dim, group_dim, current_index, scale_range, function)

        # Grouping
        coef, Feature_names = help.not_zero_feature(Lasso_model.coef_,
                                                        help.feature_names_normalization(Feature_names))

        groups = help.group_DFS(group_dim, Feature_names, max_variables_num)
        # print(groups)
        bias = current_index * group_dim
        for g in groups:
            for i in range(len(g)):
                g[i] += bias

        for g in groups:
            if not g or g is None:
                groups.remove(g)
        # We need to check the relationship between new groups and previous groups

        temp_groups = []
        for i in range(len(All_groups)):
            for j in range(len(groups)):
                if i < len(All_groups) and j < len(groups):
                    verify_time += 1

                    if not help.Differential(All_groups[i][0], groups[j][0], function, intercept, one_bias):
                        g1 = All_groups.pop(i)
                        g2 = groups.pop(j)
                        temp_groups.append(g1 + g2)
                        i -= 1
                        j -= 1
                        break

        for g in All_groups:
            temp_groups.append(g)
        for g in groups:
            temp_groups.append(g)
        All_groups = temp_groups.copy()

    return All_groups, verify_time + 100000


def CCDE(Dim):
    groups = []
    for i in range(Dim):
        groups.append([i])
    return groups


def Normal(Dim=1000):
    group = []
    for i in range(Dim):
        group.append(i)
    return [group]


def DECC_G(Dim, groups_num=10, max_number=100):
    groups = []
    for i in range(groups_num):
        groups.append([])
    for i in range(Dim):
        index = random.randint(0, groups_num-1)
        while len(groups[index]) >= max_number:
            index = random.randint(0, groups_num - 1)
        groups[index].append(i)
    # print(groups)
    return groups


def k_s(groups_num=10, max_number=100):
    groups = []
    for i in range(groups_num):
        group = list(range(i*max_number, (i+1)*max_number, 1))
        groups.append(group)
    return groups


def DECC_D(func_num, groups_num=10, max_number=100):

    bench = Benchmark()
    function = bench.get_function(func_num)
    benchmark_summary = bench.get_info(func_num)
    scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]
    Dim = 1000
    groups = k_s(10, max_number)
    delta = [0] * Dim
    NIND = 10000
    for i in range(len(groups)):
        delta[i*max_number:(i+1)*max_number] = DE.OptTool(Dim, NIND, 1, function, groups[i], scale_range, -1)
    sort_index = np.argsort(delta).tolist()
    groups = []
    for i in range(groups_num):
        groups.append(sort_index[i*max_number:(i+1)*max_number])
    return groups


def DECC_DG(func_num):
    cost = 2
    bench = Benchmark()
    function = bench.get_function(func_num)
    groups = CCDE(1000)
    intercept = function(np.zeros((1, 1000))[0])

    for i in range(len(groups)-1):
        if i < len(groups) - 1:
            cost += 2
            index1 = np.zeros((1, 1000))[0]
            index1[groups[i][0]] = 1
            delta1 = function(index1) - intercept

            for j in range(i+1, len(groups)):
                cost += 2
                if i < len(groups)-1 and j < len(groups) and not help.DG_Differential(groups[i][0], groups[j][0], delta1, function, intercept):
                    groups[i].extend(groups.pop(j))
                    j -= 1

    return groups, cost
