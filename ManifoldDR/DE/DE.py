from ManifoldDR.DE import MyProblem, templet
from ManifoldDR.util import help
from ManifoldDR.model import PolynomialModel
import geatpy as ea
import numpy as np
import umap


def CC(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    initial_Population = help.initial_population(NIND, groups, [scale_range[1]]*Dim, [scale_range[0]]*Dim, based_population)
    ave_dim = 0
    k_neighbor = 3
    for group in groups:
        ave_dim += len(group)
    ave_dim = int(ave_dim / len(groups))
    for i in range(len(groups)):
        real_iteration = 0
        while real_iteration < MAX_iteration:
            # if real_iteration == 1 and len(groups[i]) > 20:
            if real_iteration > 0 and real_iteration % 20 == 0 and len(groups[i]) > ave_dim and len(groups[i]) > 10:
                degree = 4
                UMap = umap.UMAP(n_components=3)
                data = UMap.fit_transform(initial_Population[i].Chrom)

                label = initial_Population[i].ObjV
                model = PolynomialModel.Regression(degree, data, label)
                up = []
                down = []
                for t in range(len(data[0])):
                    up.append(max(data[:, t]))
                    down.append(min(data[:, t]))

                # Surrogate model optimization
                model_var_trace, model_obj_trace, model_population = model_Optimization(degree, 100, model, up, down, data)

                # model_data = UMap.inverse_transform(model_population.Chrom)
                model_data = help.able_inverse_simulate(UMap, initial_Population[i].Chrom, data, model_population.Chrom,
                                                        np.argmin(model_obj_trace), k=k_neighbor)

                # Real problem optimization
                var_trace, obj_trace, function_population = CC_Optimization(1, benchmark, scale_range, groups[i],
                                                                              based_population, initial_Population[i],
                                                                              real_iteration)

                model_filled_data = help.filling(Dim, model_data, groups[i])
                obj_model = []
                for d in model_filled_data:
                    obj_model.append(benchmark(d))

                function_filled_data = help.filling(Dim, function_population.Chrom, groups[i])
                obj_function = []
                for d in function_filled_data:
                    obj_function.append(benchmark(d))

                initial_Population[i].Chrom, initial_Population[i].ObjV = help.find_n_best(np.vstack((model_data, function_population.Chrom))
                                                                                           , np.array(obj_model + obj_function),
                                                                                           len(model_data))

                for element in groups[i]:
                    var_traces[real_iteration, element] = initial_Population[i].Chrom[0][groups[i].index(element)]
                    based_population[element] = initial_Population[i].Chrom[0][groups[i].index(element)]

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

    return var_trace, obj_trace[:, 1], population


