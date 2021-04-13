import numpy as np


def find_n_best(Chroms, ObjVs, n):
    new_Chroms = []
    new_ObjVs = []
    index = ObjVs.argsort()
    for i in range(n):
        new_Chroms.append(Chroms[index[i]])
        new_ObjVs.append(ObjVs[index[i]])
    return np.array(new_Chroms), np.array(new_ObjVs)


Chroms = np.array([
    [1,1,1],
    [2,2,2],
    [3,3,3],
    [4,4,4],
    [5,5,5]
])

ObjVs = np.array([
    5,4,3,2,1
])

print(find_n_best(Chroms, ObjVs, 3))