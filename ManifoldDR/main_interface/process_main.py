import numpy as np
from ManifoldDR.util import help
from scipy.stats import kruskal, mannwhitneyu
import matplotlib.pyplot as plt

def ave(array):
    result = []
    for i in range(len(array[0])):
        result.append(np.mean(array[:, i]))
    return result


def final(DECC_L, DECC_LM):
    DECC_L_final = DECC_L[:, len(DECC_L[0])-1]
    DECC_LM_final = DECC_LM[:, len(DECC_LM[0]) - 1]

    print('DECC_L final: ', '%e' % np.mean(DECC_L_final), '±', '%e' % np.std(DECC_L_final, ddof=1))
    print('DECC_LM final: ', '%e' % np.mean(DECC_LM_final), '±', '%e' % np.std(DECC_LM_final, ddof=1))


def Normalization(Matrix):
    max_len = 0
    for v in Matrix:
        max_len = max(max_len, len(v))
    for v in Matrix:
        while len(v) < max_len:
            value = v[0]
            v.insert(0, value)
    return Matrix


def draw_summary(x_DECC_L, x_DECC_LM, DECC_L_ave, DECC_LM_ave):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.semilogy(x_DECC_L, DECC_L_ave, label='従来法2', linestyle=':')
    plt.semilogy(x_DECC_LM, DECC_LM_ave, label='提案法2', color='red')

    font_title = {'size': 18}
    font = {'size': 16}
    plt.title('$f_{6}$', font_title)
    plt.xlabel('Fitness evaluation times (×${10^6}$)', font)
    plt.ylabel('Fitness', font)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    DECC_L = [[1081117.2749839386, 1081117.2749839386, 1081117.2749839386, 1081117.2749839386, 1081117.2749839386, 1081117.2749839386, 1081117.2749839386, 1081117.2749839386, 1081117.2749839386, 1081117.2749839386, 1081117.2749839386, 1079616.6047988543, 1079616.6047988543, 1079616.6047988543, 1071582.329988695, 1071582.3234022476, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1071582.3210774488, 1063324.1841752084, 1063324.1808378687],
[1085619.84026742, 1080334.2185689458, 1080334.2185689458, 1077343.239229868, 1077343.2342978227, 1077343.2342978227, 1077343.2342978227, 1077343.2342978227, 1077343.2342978227, 1077343.2342978227, 1077343.2342978227, 1077343.2342978227, 1077343.2342978227, 1077343.2342978227, 1077343.2342978227, 1077343.2342978227, 1074687.3352977762, 1074687.3311065037, 1074687.3228230744, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1074687.2860612695, 1064207.4422323192, 1064207.4373205018],
[1083666.4477689464, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079360.224023503, 1079270.0520307356, 1079270.0520307356, 1079270.050158813, 1079270.0470884035, 1079270.0397572292, 1079270.0309188708, 1079270.0309188708, 1079270.0309188708, 1075279.5614589194, 1075279.5605889196, 1075279.5589743224, 1075279.5589743224, 1075279.5589743224, 1075279.5589743224, 1075279.5589743224, 1075279.5589743224, 1075279.5589743224, 1075279.5589743224, 1075279.5589743224, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1075277.2492974743, 1063596.3775374603, 1063596.3735768714, 1063596.3681333675, 1063596.3661719356, 1063596.3625660052, 1063596.3623130885, 1063596.3539436373, 1063595.837622957],
[1078956.9688762901, 1078956.9688762901, 1078956.9688762901, 1078956.9688762901, 1074384.3515812159, 1074384.3515812159, 1074384.3515812159, 1074384.3515812159, 1074384.3515812159, 1074384.3515812159, 1074384.3515812159, 1074384.3515812159, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1071455.7071062657, 1065239.0674581614, 1064657.0398281792],
[1083454.568617223, 1083454.568617223, 1083454.568617223, 1079529.8253768664, 1079529.8146655094, 1079529.8146655094, 1079529.8146655094, 1079529.8146655094, 1079529.8146655094, 1079529.8146655094, 1079529.8146655094, 1079529.8146655094, 1079529.8146655094, 1077475.5512230713, 1077475.5489800607, 1077475.5489800607, 1077475.5489800607, 1077475.5489800607, 1077475.5489800607, 1077475.5489800607, 1077475.5489800607, 1077475.5489800607, 1077475.5489800607, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1076492.5881394085, 1075958.7759596917, 1075958.7677346212, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1075958.7622915108, 1064532.451634211, 1064532.449366158, 1064532.4370775025, 1064519.0804011, 1064519.0781403156, 1064519.0772067008, 1064518.8006226264, 1064518.795185889]
]

    DECC_LM = [[1083260.761583839, 1083260.755332316, 1083260.7432783095, 1083260.7432783095, 1083260.7381326084, 1083260.7381326084, 1082518.423444669, 1082518.4186756124, 1082518.41206611, 1082518.4102031186, 1082518.4102031186, 1082518.4102031186, 1082518.4102031186, 1082518.4102031186, 1082518.4102031186, 1082518.4102031186, 1082518.4102031186, 1082518.4102031186, 1082518.4102031186, 1082518.4102031186, 1082518.4102031186, 1079182.173293425, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1079182.1712813722, 1059585.8724095651, 1059585.863463961, 1059585.8617384043, 1059585.8529296059, 1059585.8523013894, 1059585.8441549884, 1059585.8423893421, 1059460.5284466925, 1059460.5231159795, 1059460.5223038585, 1059460.5201069864, 1059460.516473268, 1059460.5139685783, 1059460.5139685783, 1059460.5079977703, 1059460.5050094503, 1059460.503203415, 1059460.5024234657, 1059460.499886392, 1059460.499886392, 1059460.4987268043, 1059460.4987268043, 1059460.4959061176, 1059460.491228626, 1059460.4832587112, 1059460.4790060283, 1059460.4752145743, 1059460.4747385576, 1059460.4747385576, 1059460.4724569293, 1059460.179649094, 1059460.1736724917, 1059460.1736724917, 1059460.1736724917, 1059460.1736724917, 1059460.1736724917, 1059460.1736724917, 1059366.9269070204],
[1069471.8065347753, 1065300.5350839372, 1065300.5267890322, 1065285.9240233905, 1065285.921953177, 1065285.9184894138, 1065285.9178843892, 1065285.910086748, 1065285.9094470574, 1065285.9049380955, 1065285.9041214641, 1065285.9041214641, 1065285.9041214641, 1065285.9037476005, 1065285.8900274476, 1065285.8877052625, 1065229.8891021577, 1065229.8790713395, 1065229.8734847424, 1065229.8149491013, 1065229.8115635475, 1065229.8115635475, 1065229.8078092139, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1065229.802511747, 1061158.3178542994, 1061158.3103791291, 1061158.2960659785, 1061158.291079129, 1061158.2869911066, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061158.2823950285, 1061111.4770275825, 1061111.4740680747, 1061111.4688168573, 1061111.4632525917, 1061111.450723559],
[1068695.3391671597, 1066570.3372449784, 1066570.3331028938, 1066570.32126478, 1066437.581631406, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1066437.5771881582, 1062933.060397762, 1062933.0555123799, 1062933.0555123799, 1062933.0534668632, 1062933.045720312, 1062933.043820856, 1062933.039866754, 1062933.0368896052, 1062933.0367131475, 1062933.0367131475, 1062933.0367131475, 1062933.0367131475, 1062254.8434835498, 1062254.8413044643, 1062254.824959518, 1062254.824959518, 1062254.823921526, 1062254.8230785383, 1062254.8165807466, 1062254.8165807466, 1062254.8165807466, 1062254.8165807466, 1062254.8165807466, 1062254.6120002053, 1062254.6108390833, 1062254.608817498],
[1066927.8787448094, 1066927.8722459995, 1066927.8718290366, 1066927.8624949807, 1066927.859654788, 1066927.8571360721, 1066927.8520976375, 1066927.8475929883, 1066927.8433121564, 1066826.7188198946, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1066826.713924075, 1062717.8335547168, 1062717.826652895, 1062717.826652895, 1062717.826652895, 1062717.826652895, 1062717.826652895, 1062717.826652895, 1062717.6144729808, 1062717.6119001885, 1062717.6087317269, 1062717.598406608, 1062717.595188128, 1062717.5933599852, 1062717.5890487803, 1062717.5890487803, 1062717.5890487803, 1062717.5888787368, 1062717.5888787368, 1062717.5882730673, 1062717.5877476435, 1062717.5866796977, 1062717.582386492, 1062717.5814536475, 1062717.5734724146, 1062717.5695195377, 1062616.589174729, 1062616.5812564667, 1062616.5734977599, 1062616.5681267926, 1062616.5647606004, 1062616.5612627107, 1062616.5608407028, 1062616.5594907864, 1062616.5574482915, 1062616.540360756, 1062616.5338183567],
[1067893.8669800158, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067893.8634364677, 1067279.866290837, 1067279.8602878063, 1067279.857209791, 1067279.8558100432, 1067279.8529192526, 1067279.8485704022, 1067279.8428776774, 1067279.841325794, 1067279.841325794, 1067279.841325794, 1067011.2308049328, 1067011.22766312, 1067011.2246682015, 1067011.2149554044, 1067011.2091065794, 1067011.2028612155, 1067011.1989100408, 1067011.1947948795, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1067011.1898030888, 1064255.4069573674, 1064255.3936505914, 1064255.381230384, 1064255.381230384, 1064255.3793780277, 1064255.3709572125, 1064255.3687463084, 1064255.3671648034, 1064255.363053096, 1064238.1438133053, 1064238.1423991965, 1064238.1419518588, 1064238.1419466536, 1064238.1408907557, 1064238.1377230138, 1064238.1376852246, 1064238.1315953506, 1064238.1275659462, 1060985.3818701038, 1060985.372878007, 1060985.372878007, 1060985.3619743795, 1060985.3580070047, 1060890.1148126605, 1060890.1095075915, 1060890.0731510017, 1060890.0711278748, 1060890.0701013613, 1060890.0625081812, 1060890.0625081812, 1060890.0617154383, 1060890.0599082066, 1060890.058230714, 1060890.0555921816, 1060890.0551281704, 1060890.0482986122, 1060890.0482986122, 1060890.0482986122, 1060890.0482986122, 1060890.0482986122, 1060890.0482986122]

]

    DECC_L = np.array(Normalization(DECC_L))
    DECC_LM = np.array(Normalization(DECC_LM))

    final(DECC_L, DECC_LM)

    DECC_L_ave = ave(DECC_L)
    DECC_LM_ave = ave(DECC_LM)

    # Normal/One/DECC-G use same x
    LASSO_cost = 0

    bias = 30000

    x_DECC_L = np.linspace(LASSO_cost+bias, 3000000, len(DECC_L_ave))
    x_DECC_LM = np.linspace(LASSO_cost+bias, 3000000, len(DECC_LM_ave))

    draw_summary(x_DECC_L, x_DECC_LM, DECC_L_ave, DECC_LM_ave)

    # print('kruskal: ', kruskal(DECC_L[:, len(DECC_L[0])-1], Normal[:, len(Normal[0])-1],
    #               One[:, len(One[0])-1], DECC_DG[:, len(DECC_DG[0])-1], DECC_D[:, len(DECC_D[0])-1],
    #               DECC_G[:, len(DECC_G[0])-1]))
    #
    # print('Holm test: ', Holm(DECC_L[:, len(DECC_L[0])-1], Normal[:, len(Normal[0])-1],
    #               One[:, len(One[0])-1], DECC_DG[:, len(DECC_DG[0])-1], DECC_D[:, len(DECC_D[0])-1],
    #               DECC_G[:, len(DECC_G[0])-1], mannwhitneyu))
