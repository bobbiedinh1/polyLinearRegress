import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#get dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#replace missing data
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# X[:, 1] = imputer.fit_transform(X[:, 1]) 
#or you can fit (learn from the model first) then transform (change the model))

# #encode catagorical data
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
# remember columns do not return as a numpy array so remember to np.array() for conversion


# #encode dependent catagorical data
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()

#create training set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# #feature scaling
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
newPosition = poly_reg.fit_transform([[6.5]])

np.set_printoptions(precision=2)
print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(newPosition))

#creating graphs 
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,5))
# axes[0].scatter(X,y, color='red')
# axes[0].plot(X, lin_reg.predict(X), color = 'blue')
# axes[0].set_title('Truth or Bluff')
# axes[0].set_xlabel('Position')
# axes[0].set_ylabel('Salary')

# axes[1].scatter(X, y, color ='red')
# axes[1].plot(X, lin_reg_2.predict(X_poly), color = 'blue')
# axes[1].set_title('Truth or Bluff')
# axes[1].set_xlabel('Position')
# axes[1].set_ylabel('Salary')
# plt.show()

#when we use grid method we get a cleaner curve, we draw the line using points such as 1.1, 1.2, 1.3... instead of 1, 2 ,3 
#hence min max and 0.1 in the arrange method
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue', label='Linear Regression')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='green', label='Polynomial Regression (degree=4)')
plt.title('Truth or Bluff')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.legend()
plt.show()