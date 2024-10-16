import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:, 1:-1].values
Y = data.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
model_Lin= LinearRegression()
model_Lin.fit(X, Y)

plt.scatter(X, Y, color = 'red')
plt.plot(X, model_Lin.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
model_Poly = PolynomialFeatures(degree = 4)
X_poly = model_Poly.fit_transform(X)
model_Lin_2 = LinearRegression()
model_Lin_2.fit(X_poly, Y)

plt.scatter(X, Y, color = 'red')
plt.plot(X, model_Lin_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()