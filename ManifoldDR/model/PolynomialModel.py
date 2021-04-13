from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


def Regression(degree, train_data, train_label):

    poly_reg = PolynomialFeatures(degree=degree)

    train_data_ploy = poly_reg.fit_transform(train_data)
    reg_Polynomial = linear_model.LinearRegression()
    reg_Polynomial.fit(train_data_ploy, train_label)

    return reg_Polynomial
