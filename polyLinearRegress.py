import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#get dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, -1].values
X = X.reshape(-1,1)

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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# #feature scaling
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y.reshape(len(y),1)), axis=1))




#creating graphs 
fig, axes = plt.subplots(nrows=1, ncols=4, figsize = (12,5))