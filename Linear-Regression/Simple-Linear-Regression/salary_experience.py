#first machine learning program :]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data preprocessing

#loading data
dataset=pd.read_csv("Salary_Data.csv")
#independent variables matrix
x=dataset.iloc[:, :-1].values
#dependent variable vector
y=dataset.iloc[:, 1].values
#spliting the training and testing data.
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=0.33, random_state=42)

#traning model

#importing Linear Regression model
from sklearn.linear_model import LinearRegression
#training model
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
#predicting results
ypred=regressor.predict(xtest)

#visualizing the predictions
plt.scatter(xtest, ytest, color="red")
plt.plot(xtest, ypred, color="purple")
plt.title("Salary x Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
