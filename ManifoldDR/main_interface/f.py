from ManifoldDR.util import help
from ManifoldDR.DE import DE
from cec2013lsgo.cec2013 import Benchmark


def CC_exe(Dim, func_num, NIND, Max_iteration, scale_range, groups, method):
    bench = Benchmark()
    function = bench.get_function(func_num)
    name = 'f' + str(func_num)
    print(name, 'Optimization with', method)

    """The next is DE optimization"""
    best_indexes, best_obj_trace = DE.CC(Dim, NIND, Max_iteration, function, scale_range, groups)
    help.write_obj_trace(name, method, best_obj_trace)


