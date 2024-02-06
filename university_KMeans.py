# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:01:06 2023

@author: ASUS
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

km = KMeans()
data = pd.read_excel('University_Clustering.xlsx')
data

data.describe()
data = data.drop(['State'], axis = 1)
data


def norm_funct(i):
    x = (i- i.min())/(i.max()-i.min())
    return x


df_norm = norm_funct(data.iloc[:,1:])
df_norm

ESS = []
k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    ESS.append(kmeans.inertia_)
    
ESS


plt.plot(k, ESS, 'ro-') 
plt.xlabel('No_Of_Clusters')
plt.ylabel("Total_within_ss")



model = KMeans(n_clusters=3)
model.fit(df_norm)
#x = model.labels_
mb = pd.Series(model.labels_)
mb
data['Clust'] = mb
data['clust'] = model.labels_
data

data = data.iloc[:,[7,0,1,2,3,4,5,6,]]
data

data.iloc[:,2:8].groupby(data.Clust).mean()


d1 = data[data.Clust == 0]
d2 = data[data.Clust == 1]
d3 = data[data.Clust == 2]

plt.scatter(d1['SAT'], d1['Accept'], color = 'Green')
plt.scatter(d2['SAT'], d2['Accept'], color = 'Red')
plt.scatter(d3['SAT'], d3['Accept'], color = 'Blue')
#plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color = 'purple', marker="*", label = 'centroid')



data.to_csv('KMeans_university_clustering.csv', encoding = 'utf-8')
import os
os.getcwd()

















