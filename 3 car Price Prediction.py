# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:32:48 2023

@author: Aishwarya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv")
dataset['CarName'], dataset['symboling']=dataset.iloc[:, 1].values, dataset.iloc[:,2].values
col_list = list(dataset)
col_list[1],col_list[2] = col_list[2], col_list[1]
dataset.columns = col_list
dataset
dataset.describe()
dataset.isnull().sum()
dataset.info()
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(dataset.price)
plt.show()
x1 = dataset.fueltype.unique()
mp1 = dict(zip(x1, range(len(x1))))
x2 = dataset.aspiration.unique()
mp2 = dict(zip(x2, range(len(x2))))
x3 = dataset.doornumber.unique()
mp3 = dict(zip(x3, range(len(x3))))
x4 = dataset.carbody.unique()
mp4 = dict(zip(x4, range(len(x4))))
x5 = dataset.drivewheel.unique()
mp5 = dict(zip(x5, range(len(x5))))
x6 = dataset.enginelocation.unique()
mp6 = dict(zip(x6, range(len(x6))))
x7 = dataset.fuelsystem.unique()
mp7 = dict(zip(x7, range(len(x7))))
x8 = dataset.enginetype.unique()
mp8 = dict(zip(x8, range(len(x8))))
x9 = dataset.cylindernumber.unique()
mp9 = dict(zip((x9), range(len(x9))))
dataset = dataset.replace({'fueltype': mp1, 'aspiration':mp2, 'doornumber':mp3, 'carbody':mp4,'drivewheel':mp5,
                           'enginelocation':mp6, 'fuelsystem':mp7, 'enginetype':mp8, 'cylindernumber':mp9})
dataset
dataset['CarName']
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values
X
y

#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Training a Car Price Prediction Model
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)


#Prediction
y_pred = model.predict(X_test)
y_pred


#Accuracy
from sklearn.metrics import mean_absolute_error
model.score(X_test, y_pred)
