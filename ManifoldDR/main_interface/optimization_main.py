from ManifoldDR.main_interface import f
from ManifoldDR.util import help
from ManifoldDR.main_interface.grouping_main import LASSOCC, DECC_DG, DECC_D, DECC_G, CCDE, Normal
from cec2013lsgo.cec2013 import Benchmark
import time


if __name__ == '__main__':

    Dim = 1000
    NIND = 30
    bench = Benchmark()
    EFs = 3000000
    for func_num in range(1, 16):
        test_time = 1
        name = 'f' + str(func_num)
        benchmark_summary = bench.get_info(func_num)
        scale_range = [benchmark_summary['lower'], benchmark_summary['upper']]

        for i in range(test_time):
            # LASSO_Groups, LASSO_cost = LASSOCC(func_num)
            LASSO_Groups = DECC_G(Dim, 10, 100)
            LASSO_cost = 0
            f.CC_exe(Dim, func_num, NIND, int((EFs - LASSO_cost)/(NIND * Dim)), scale_range, LASSO_Groups, 'DECC_LM')

            print('    Finished: ', 'function: ', func_num, 'iteration: ', i + 1, '/', test_time)


