import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predict = model.predict(diabetes_X_test)
print(diabetes_Y_predict)

print("MSE", mean_squared_error(diabetes_Y_test, diabetes_Y_predict))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)


# plt.scatter(diabetes_X_test, diabetes_Y_test)
# plt.plot(diabetes_X_test, diabetes_Y_predict)
# plt.show()


# Stats considering one of the feature
# diabetes_X = diabetes.data[:, np.newaxis, 2]
# MSE 3035.0601152912686
# Weights:  [941.43097333]
# Intercept:  153.39713623331698

# Stats considering all the features
# MSE 1826.5364191345423
# Weights:  [  -1.16924976 -237.18461486  518.30606657  309.04865826 -763.14121622
#   458.90999325   80.62441437  174.32183366  721.49712065   79.19307944]
# Intercept:  153.05827988224112