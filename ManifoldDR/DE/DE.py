from ManifoldDR.DE import MyProblem, templet
from ManifoldDR.util import help
from ManifoldDR.model import PolynomialModel
import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy



# Asynchronous
def DECC_CL_CCDE(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    initial_Population = help.initial_population(NIND, groups, [scale_range[1]] * Dim, [scale_range[0]] * Dim,
                                                 based_population)
    real_iteration = 0
    Generations = []
    N = 3
    up = [scale_range[1]] * Dim
    down = [scale_range[0]] * Dim
    while real_iteration < MAX_iteration:

        if real_iteration > N and not help.is_Continue(Generations[len(Generations)-N:len(Generations)], threshold=0.1):
             break
        for i in range(len(groups)):
            var_trace, obj_trace, initial_Population[i] = CC_Optimization(1, benchmark, scale_range, groups[i],
                                                                          based_population, initial_Population[i],
                                                                          real_iteration)
            if (up[i] - down[i]) / (scale_range[1] - scale_range[0]) > 0.9:
                up[i] = max(initial_Population[i].Chrom[:, 0])
                down[i] = min(initial_Population[i].Chrom[:, 0])

            var_traces[real_iteration, groups[i][0]] = var_trace[1, 0]
            based_population[groups[i][0]] = var_trace[1, 0]
        Generations.append(benchmark(var_traces[real_iteration]))

        real_iteration += 1
    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces, real_iteration * NIND * Dim, up, down


# Asynchronous
def DECC_CL_DECC_L(Dim, NIND, MAX_iteration, benchmark, up, down, groups, elite):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = copy.deepcopy(elite)
    initial_Population = help.initial_population(NIND, groups, up, down, elite)
    real_iteration = 0

    while real_iteration < MAX_iteration:
        for i in range(len(groups)):
            var_trace, obj_trace, initial_Population[i] = CC_Limit_Optimization(1, benchmark, up, down, groups[i],
                                                               based_population, initial_Population[i], real_iteration)

            for element in groups[i]:
                var_traces[real_iteration, element] = var_trace[1, groups[i].index(element)]
                based_population[element] = var_trace[1, groups[i].index(element)]
        real_iteration += 1

    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces


def OptTool(Dim, NIND, MAX_iteration, benchmark, scale_range, maxormin):
    problem = MyProblem.MyProblem(Dim, benchmark, scale_range, maxormin)  # 实例化问题对象

    """==============================种群设置==========================="""
    Encoding = 'RI'  # 编码方式
    NIND = NIND  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    population.initChrom(NIND)
    """===========================算法参数设置=========================="""

    # myAlgorithm = templet.soea_SaNSDE_templet(problem, population)
    myAlgorithm = ea.soea_DE_currentToBest_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run()
    # obj_traces.append(obj_trace[0])
    return var_trace[len(var_trace)-1]


def CC(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    initial_Population = help.initial_population(NIND, groups, [scale_range[1]]*Dim, [scale_range[0]]*Dim, based_population)

    real_iteration = 0
    while real_iteration < MAX_iteration:
        for i in range(len(groups)):
            if real_iteration % 20 == 0 and len(groups[i]) > 20:
                degree = 4
                new_Dim = int(len(groups[i]) / 5)
                pca = PCA(n_components=new_Dim)
                data = pca.fit_transform(initial_Population[i].Chrom)
                label = initial_Population[i].ObjV
                model = PolynomialModel.Regression(degree, data, label)
                up = []
                down = []
                for t in range(len(data[0])):
                    up.append(max(data[:, t]))
                    down.append(min(data[:, t]))
                # Surrogate model optimization
                var_trace, obj_trace, model_population = model_Optimization(degree, 100, model, up, down, data)

                model_data = pca.inverse_transform(model_population.Chrom)
                model_filled_data = help.filling(Dim, model_data, groups[i])
                obj_model = []
                for d in model_filled_data:
                    obj_model.append(benchmark(d))
                # Real problem optimization

                var_trace, obj_trace, function_population = CC_Optimization(1, benchmark, scale_range, groups[i],
                                                                              based_population, initial_Population[i],
                                                                              real_iteration)
                function_filled_data = help.filling(Dim, function_population.Chrom, groups[i])
                obj_function = []
                for d in function_filled_data:
                    obj_function.append(benchmark(d))

                initial_Population[i].Chrom, initial_Population[i].ObjV = help.find_n_best(model_data +
                                                                                           function_population.Chrom,
                                                                                           obj_model + obj_function,
                                                                                           len(model_data))
                for element in groups[i]:
                    var_traces[real_iteration, element] = var_trace[1, groups[i].index(element)]
                    based_population[element] = var_trace[1, groups[i].index(element)]


            else:
                var_trace, obj_trace, initial_Population[i] = CC_Optimization(1, benchmark, scale_range, groups[i],
                                                        based_population, initial_Population[i], real_iteration)
                for element in groups[i]:
                    var_traces[real_iteration, element] = var_trace[1, groups[i].index(element)]
                    based_population[element] = var_trace[1, groups[i].index(element)]
        real_iteration += 1

    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    x = np.linspace(0, 3000000, len(obj_traces))
    help.draw_check(x, obj_traces, 'check')
    return var_traces, obj_traces


def CC_Optimization(MAX_iteration, benchmark, scale_range, group, based_population, p, real):
    problem = MyProblem.CC_Problem(group, benchmark, scale_range, based_population)  # 实例化问题对象

    """===========================算法参数设置=========================="""

    myAlgorithm = templet.soea_DE_currentToBest_1_L_templet(problem, p)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run(real)
    # obj_traces.append(obj_trace[0])

    return var_trace, obj_trace, population


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

    return var_trace, obj_trace, population


def CC_Limit_Optimization(MAX_iteration, benchmark, up, down, group, based_population, p, real):
    problem = MyProblem.Block_Problem(group, benchmark, up, down, based_population)  # 实例化问题对象

    """===========================算法参数设置=========================="""

    myAlgorithm = templet.soea_DE_currentToBest_1_L_templet(problem, p)
    myAlgorithm.MAXGEN = MAX_iteration
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    # [population, obj_trace, var_trace] = myAlgorithm.run(population, MAX_iteration)
    [population, obj_trace, var_trace] = myAlgorithm.run(real)
    # obj_traces.append(obj_trace[0])
    return var_trace, obj_trace, population

