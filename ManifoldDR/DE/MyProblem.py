import geatpy as ea
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class CC_Problem(ea.Problem):
    def __init__(self, group, benchmark, scale_range, based_population):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
        self.Dim = len(group)
        varTypes = [0] * self.Dim
        lb = [scale_range[0]] * self.Dim
        ub = [scale_range[1]] * self.Dim
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        self.benchmark = benchmark
        self.group = group
        self.based_population = based_population
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象

        temp_Phen = []
        for i in range(len(pop.Chrom)):
            temp_Phen.append(self.based_population)
        temp_Phen = np.array(temp_Phen)

        for element in self.group:
            temp_Phen[:, element] = pop.Phen[:, self.group.index(element)]

        result = []
        for p in temp_Phen:
            result.append([self.benchmark(p)])
        pop.ObjV = np.array(result)
        # print(min(result))


class Block_Problem(ea.Problem):
    def __init__(self, group, benchmark, up, down, based_population):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
        self.Dim = len(group)
        varTypes = [0] * self.Dim
        sub_lb = []
        sub_ub = []
        for e in group:
            sub_lb.append(down[e])
            sub_ub.append(up[e])
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        self.benchmark = benchmark
        self.group = group
        self.based_population = based_population

        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, sub_lb, sub_ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        temp_Phen = []
        for i in range(len(pop.Chrom)):
            temp_Phen.append(self.based_population)
        temp_Phen = np.array(temp_Phen)

        for element in self.group:
            temp_Phen[:, element] = pop.Phen[:, self.group.index(element)]

        result = []
        for p in temp_Phen:
            result.append([self.benchmark(p)])
        pop.ObjV = np.array(result)
        # print(min(pop.ObjV))


class MyProblem(ea.Problem):
    def __init__(self, Dim, benchmark, scale_range, maxormin):
        name = 'MyProblem'
        M = 1
        self.Dim = Dim
        self.benchmark = benchmark
        maxormins = [maxormin]
        varTypes = [0] * self.Dim
        lb = [scale_range[0]] * self.Dim
        ub = [scale_range[1]] * self.Dim
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        result = []
        for p in pop.Phen:
            result.append([self.benchmark(p)])
        pop.ObjV = np.array(result)


class modelProblem(ea.Problem):
    def __init__(self, degree, Dim, model, up, down):
        name = 'MyProblem'
        M = 1
        self.Dim = Dim
        self.model = model
        self.degree = degree
        maxormins = [1]
        varTypes = [0] * self.Dim
        lb = down
        ub = up
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        poly_reg = PolynomialFeatures(degree=self.degree)
        pop.ObjV = self.model.predict(poly_reg.fit_transform(pop.Chrom))


class distanceProblem(ea.Problem):
    def __init__(self, Dim, loss, up, down, low_D_k_Normal_dis, k_index, high_D_origin_pop):
        name = 'MyProblem'
        M = 1
        self.Dim = Dim
        self.loss = loss
        self.low_D_k_Normal_dis = low_D_k_Normal_dis
        self.k_index = k_index
        self.high_D_origin_pop = high_D_origin_pop
        maxormins = [1]
        varTypes = [0] * self.Dim
        lb = down
        ub = up
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        result = []
        for Chrom in pop.Chrom:
            # print(Chrom)
            result.append([self.loss(self.low_D_k_Normal_dis, self.k_index, self.high_D_origin_pop, Chrom)])
        pop.ObjV = np.array(result)

