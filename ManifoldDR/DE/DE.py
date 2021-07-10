from ManifoldDR.DE import MyProblem, templet
from ManifoldDR.model.PolynomialModel import Regression
from ManifoldDR.util import help
import numpy as np
import geatpy as ea
import umap
import warnings


def CC_LM(Dim, NIND, MAX_iteration, benchmark, scale_range, groups):
    warnings.filterwarnings("ignore")
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

        # print(len(groups[i]), ave_dim)
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

                better_data, better_z, worse_data, worse_z = help.data_split(data, fitness)
                low_D_data = UMap.fit_transform(data)

                up = [max(low_D_data[:, 0]), max(low_D_data[:, 1])]
                down = [min(low_D_data[:, 0]), min(low_D_data[:, 1])]

                # Surrogate model building & optimization & data restore depending on worse data

                degree = 4
                reg = Regression(degree, low_D_data, fitness)

                model_Population = model_Opt(2, degree, reg, up, down, len(worse_z))
                model_data = UMap.inverse_transform(model_Population.Chrom)

                # Real problem optimization
                """=================算法模板参数设定============================"""

                # Define the new population for separate Optimization
                real_Population = real_Opt(groups[i], benchmark, scale_range, based_population, better_data, better_z)

                # Fill the chrom and evaluate
                model_filled_data, obj_model = help.filling(model_data, groups[i], based_population, benchmark)

                Chrom, ObjV = help.find_n_best(np.vstack((worse_data, model_data)), np.append(worse_z, obj_model),
                                               len(worse_z))
                initial_Population[i].Chrom = np.vstack((Chrom, real_Population.Chrom))
                initial_Population[i].Phen = np.vstack((Chrom, real_Population.Chrom))
                initial_Population[i].ObjV = np.append(ObjV, real_Population.ObjV[:, 0]).reshape(-1, 1)

                best_index = np.argmin(initial_Population[i].ObjV[:, 0])

                for element in groups[i]:
                    var_traces[real_iteration, element] = initial_Population[i].Chrom[best_index][
                            groups[i].index(element)]
                    based_population[element] = initial_Population[i].Chrom[best_index][groups[i].index(element)]
                initial_Population[i].shuffle()

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


def model_Opt(Dim, degree, model, up, down, NIND):
    """=================算法模板参数设定============================"""
    problem = MyProblem.modelProblem(degree, Dim, model, up, down)

    Field = ea.crtfld('RI', problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population('RI', Field, NIND)
    population.initChrom(NIND)

    Algorithm = templet.soea_DE_currentToBest_1_L_templet(problem, population)
    Algorithm.MAXGEN = 50
    Algorithm.drawing = 0
    model_Population, obj_trace, var_trace = Algorithm.run()
    return model_Population


def real_Opt(group, benchmark, scale_range, based_population, data, fitness):
    problem = MyProblem.CC_Problem(group, benchmark, scale_range, based_population)
    Field = ea.crtfld('RI', problem.varTypes, problem.ranges, problem.borders)
    temp_Population = ea.Population('RI', Field, len(data))

    temp_Population.initChrom(len(data))

    temp_Population.Chrom = data
    temp_Population.Phen = data
    temp_Population.ObjV = fitness.reshape(-1, 1)

    # Execute the optimization
    Algorithm = templet.soea_DE_currentToBest_1_L_templet(problem, temp_Population)
    Algorithm.drawing = 0
    Algorithm.MAXGEN = 1
    temp_Population, obj_trace, var_trace = Algorithm.run()
    return temp_Population
