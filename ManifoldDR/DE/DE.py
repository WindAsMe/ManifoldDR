from ManifoldDR.DE import MyProblem, templet
from ManifoldDR.model.PolynomialModel import Regression
from ManifoldDR.util import help
import numpy as np
import geatpy as ea
import umap


def CC_LM(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration, Dim))
    based_population = np.zeros(Dim)
    initial_Population = help.initial_population(NIND, groups, [scale_range[1]] * Dim, [scale_range[0]] * Dim,
                                                 based_population)

    ave_dim = 0
    for group in groups:
        ave_dim += len(group)
    ave_dim /= len(groups)

    for i in range(len(groups)):
        real_iteration = 0

        """=================算法模板参数设定============================"""
        problem = MyProblem.CC_Problem(groups[i], benchmark, scale_range, based_population)
        Algorithm = templet.soea_DE_currentToBest_1_L_templet(problem, initial_Population[i])
        Algorithm.call_aimFunc(initial_Population[i])

        while real_iteration < MAX_iteration:
            if len(groups[i]) > 3 and len(groups[i]) >= ave_dim and real_iteration % 5 == 1:

                UMap = umap.UMAP(n_components=2)
                data = initial_Population[i].Chrom
                fitness = initial_Population[i].ObjV[:, 0]

                # Worse data is to build Krige model
                # Better data is to participate DE
                better_data, better_z, worse_data, worse_z = help.data_split(data, fitness, 0.5)

                low_D_data = UMap.fit_transform(data)

                # x_range = [min(low_D_data[:, 0]), max(low_D_data[:, 0])]
                # y_range = [min(low_D_data[:, 1]), max(low_D_data[:, 1])]
                up = [max(low_D_data[:, 0]), max(low_D_data[:, 1])]
                down = [min(low_D_data[:, 0]), min(low_D_data[:, 1])]
                # interval = 100
                # gridx = np.linspace(x_range[0], x_range[1], interval)
                # gridy = np.linspace(y_range[0], y_range[1], interval)

                # Surrogate model building & optimization & data restore depending on worse data
                try:
                    # k3d1 = help.Krige_model(gridx, gridy, np.array(low_D_data), fitness)
                    # indexes, best_fitness = help.find_n_matrix(k3d1, len(worse_data), gridx, gridy)
                    degree = 4
                    reg = Regression(degree, low_D_data, fitness)
                    model_Population = model_Opt(degree, reg, up, down, low_D_data, fitness)
                    indexes, best_fitness = help.find_n_matrix(model_Population.Chrom, int(len(low_D_data) / 2),
                                                               model_Population.ObjV[:, 0])

                    model_data = UMap.inverse_transform(indexes)

                    # Real problem optimization
                    """=================算法模板参数设定============================"""

                    # Define the new population for separate Optimization
                    problem = MyProblem.CC_Problem(groups[i], benchmark, scale_range, based_population)
                    Field = ea.crtfld('RI', problem.varTypes, problem.ranges, problem.borders)
                    temp_Population = ea.Population('RI', Field, NIND)

                    temp_Population.initChrom(len(better_data))

                    temp_Population.Chrom = better_data
                    temp_Population.Phen = better_data
                    temp_Population.ObjV = better_z.reshape(-1, 1)

                    # Execute the optimization
                    Algorithm = templet.soea_DE_currentToBest_1_L_templet(problem, temp_Population)
                    Algorithm.drawing = 0
                    Algorithm.MAXGEN = 1
                    temp_Population, obj_trace, var_trace = Algorithm.run()

                    # Fill the chrom and evaluate
                    model_filled_data, obj_model = help.filling(model_data, groups[i], based_population, benchmark)

                    Chrom, ObjV = help.find_n_best(np.vstack((worse_data, model_data)),
                                                    np.append(worse_z, obj_model), len(worse_data))

                    initial_Population[i].Chrom = np.vstack((Chrom, temp_Population.Chrom))
                    initial_Population[i].Phen = np.vstack((Chrom, temp_Population.Chrom))
                    initial_Population[i].ObjV = np.append(ObjV, temp_Population.ObjV[:, 0]).reshape(-1, 1)

                    best_index = np.argmin(initial_Population[i].ObjV[:, 0])

                    for element in groups[i]:
                        var_traces[real_iteration, element] = initial_Population[i].Chrom[best_index][
                            groups[i].index(element)]
                        based_population[element] = initial_Population[i].Chrom[best_index][groups[i].index(element)]
                    initial_Population[i].shuffle()

                except ValueError:
                    """=================算法模板参数设定============================"""

                    problem = MyProblem.CC_Problem(groups[i], benchmark, scale_range, based_population)
                    Algorithm = templet.soea_DE_currentToBest_1_L_templet(problem, initial_Population[i])
                    Algorithm.drawing = 0
                    Algorithm.MAXGEN = 1
                    initial_Population[i], obj_trace, var_trace = Algorithm.run()
                    # print('  Else: ', obj_trace)
                    for element in groups[i]:
                        var_traces[real_iteration, element] = var_trace[1, groups[i].index(element)]
                        based_population[element] = var_trace[1, groups[i].index(element)]

            else:

                """=================算法模板参数设定============================"""

                problem = MyProblem.CC_Problem(groups[i], benchmark, scale_range, based_population)
                Algorithm = templet.soea_DE_currentToBest_1_L_templet(problem, initial_Population[i])
                Algorithm.drawing = 0
                Algorithm.MAXGEN = 1
                initial_Population[i], obj_trace, var_trace = Algorithm.run()
                # print('  Else: ', obj_trace)
                for element in groups[i]:
                    var_traces[real_iteration, element] = var_trace[1, groups[i].index(element)]
                    based_population[element] = var_trace[1, groups[i].index(element)]
            real_iteration += 1
            # print('min objV: ', min(initial_Population[i].ObjV))

    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces


def CC_L(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    var_traces = np.zeros((MAX_iteration + 1, Dim))
    based_population = np.zeros(Dim)
    initial_Population = help.initial_population(NIND, groups, [scale_range[1]] * Dim, [scale_range[0]] * Dim,
                                                 based_population)
    for i in range(len(groups)):

        """=================算法模板参数设定============================"""

        problem = MyProblem.CC_Problem(groups[i], benchmark, scale_range, based_population)
        Algorithm = templet.soea_DE_currentToBest_1_L_templet(problem, initial_Population[i])
        Algorithm.call_aimFunc(initial_Population[i])
        Algorithm.MAXGEN = MAX_iteration
        Algorithm.drawing = 0

        initial_Population[i], obj_trace, var_trace = Algorithm.run()
        for element in groups[i]:
            var_traces[:, element] = var_trace[:, groups[i].index(element)]
            based_population[element] = var_trace[np.argmin(obj_trace[:, 1]), groups[i].index(element)]

    var_traces, obj_traces = help.preserve(var_traces, benchmark)
    return var_traces, obj_traces


def model_Opt(degree, model, up, down, data, fitness):
    """=================算法模板参数设定============================"""
    problem = MyProblem.modelProblem(degree, len(data[0]), model, up, down)

    Field = ea.crtfld('RI', problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population('RI', Field, len(data))
    population.initChrom(len(data))

    population.Chrom = np.array(data)
    population.Phen = np.array(data)
    population.ObjV = np.array(fitness).reshape(-1, 1)
    Algorithm = templet.soea_DE_currentToBest_1_L_templet(problem, population)
    Algorithm.MAXGEN = 100
    Algorithm.drawing = 0
    model_Population, obj_trace, var_trace = Algorithm.run()
    return model_Population



