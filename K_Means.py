# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:40:27 2023

@author: ASUS
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# generating 50 random no having uniform distribution in range of 0 -1

"""
unifrom distribution have the same numbers of probability of outcome for 
every number 
""" 

x = np.random.uniform(0,1,50)
y = np.random.uniform(0,1,50)

# creating an empty dataframe
df_xy = pd.DataFrame(columns=["x","y"])

df_new = pd.DataFrame()

df_xy.x = x
df_xy.y = y
df_xy


plt.scatter(x, y)


km = KMeans(n_clusters=3)     # applying KMeans model


cluster = km.fit_predict(df_xy)   # applying data to the KMeans algorithmic model
cluster

df_xy['cluster'] = cluster
df_xy

df1 = df_xy[df_xy.cluster == 0]
df2 = df_xy[df_xy.cluster == 1] 
df3 = df_xy[df_xy.cluster == 2] 

plt.scatter(df1['x'], df1['y'], color = "Green")
plt.scatter(df2['x'], df2['y'], color = "Red")
plt.scatter(df3['x'], df3['y'], color = "Blue")




