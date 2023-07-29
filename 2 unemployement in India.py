# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:24:49 2023

@author: Aishwarya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
dataset=pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
dataset.head(100)
dataset.info()
dataset.describe()
dataset.corr()
dataset['Region']
y=dataset[' Estimated Unemployment Rate (%)']
y
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,10))
sns.heatmap(dataset.corr())
plt.show()
dataset.columns= ["States","Date","Frequency",
               "Estimated Unemployment Rate","Estimated Employed",
               "Estimated Labour Participation Rate","Region",
               "longitude","latitude"]
plt.figure(figsize=(10, 8))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate", hue="Region", data=dataset)
plt.show()
sns.pairplot(dataset)

#Analyzing Data By bargraph
fg = px.bar(dataset,x='Region',y='Estimated Unemployment Rate',color='Region',title='Unemployment rate',animation_frame='Date',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()

#Analyzing Data By scatterplot¶
fg = px.scatter(dataset,x='Region',y='Estimated Unemployment Rate',color='Region',title='Unemployment rate',animation_frame='Date',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()

#Analyzing Data By Histogram¶
fg = px.histogram(dataset,x='Region',y='Estimated Unemployment Rate',color='Region',title='Unemployment rate',animation_frame='Date',template='plotly')
fg.update_layout(xaxis={'categoryorder':'total descending'})
fg.show()

#Analyzing Data by sunburst plot
unemploment = dataset[["States", "Region", "Estimated Unemployment Rate"]]
figure = px.sunburst(unemploment, path=["Region", "States"], 
                     values="Estimated Unemployment Rate", 
                     width=500, height=500, color_continuous_scale="RdY1Gn", 
                     title="Unemployment Rate in India")
figure.show()