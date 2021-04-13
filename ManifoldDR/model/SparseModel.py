from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from ManifoldDR.util import help


def get_feature_name(number):
    result = []
    for i in range(number):
        result.append(str(i))
    return result


def isSeparable(str):
    str_c = str.replace(' ', '')
    return len(str_c) == len(str)


def is_zero(coef):
    num = 0
    for i in coef:
        if i == 0:
            num += 1
    return num


def Regression(degree, train_size, total_dim, group_dim, current_index, scale_range, benchmark):

    poly_reg = PolynomialFeatures(degree=degree)
    total_train_data, real_train_data = help.create_local_model_data(train_size, total_dim, group_dim, current_index,
                                                                     scale_range)

    label = help.create_result(total_train_data, benchmark)
    real_train_data_ploy = poly_reg.fit_transform(real_train_data)

    reg_Lasso = linear_model.Lasso(max_iter=10000)
    reg_Lasso.fit(real_train_data_ploy, label)
    feature_names = poly_reg.get_feature_names(input_features=get_feature_name(group_dim))
    flag = max(abs(reg_Lasso.coef_))
    valid_feature = []
    valid_coef = []
    for i in range(len(reg_Lasso.coef_)):
        if abs(reg_Lasso.coef_[i]) > 0.1 and abs(reg_Lasso.coef_[i]) > flag * 0.1:
            valid_feature.append(feature_names[i])
            valid_coef.append(reg_Lasso.coef_[i])
            continue
        else:
            reg_Lasso.coef_[i] = 0
    # print(valid_feature)
    # print(valid_coef)
    # print('model model valid coef: ', len(reg_Lasso.coef_) - is_zero(reg_Lasso.coef_), 'intercept: ', reg_Lasso.intercept_)
    return reg_Lasso, feature_names
