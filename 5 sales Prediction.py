# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:36:39 2023

@author: Aishwarya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv("Advertising.csv")
dataset.head()
dataset.info()
dataset.describe()
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(dataset.Sales)
plt.show()
plt.style.use('seaborn-whitegrid')
sns.heatmap(dataset.corr())
plt.show()
plt.scatter(dataset['Sales'],dataset['Newspaper'])
plt.xlabel('Sales')
plt.ylabel('Newspaper')
plt.scatter(dataset['Sales'],dataset['Radio'])
plt.xlabel('Sales')
plt.ylabel('Radio')
plt.scatter(dataset['Sales'],dataset['TV'])
plt.xlabel('Sales')
plt.ylabel('TV')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred
sale = regressor.predict([[44.5,39.3,45.1]])
print(sale)
print("Score",regressor.score(X_test, y_test))
plt.scatter(y_test,y_pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');
sns.regplot(x=y_test,y=y_pred,ci=None,color ='green')