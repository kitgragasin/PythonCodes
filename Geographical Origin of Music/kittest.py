# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:12:34 2019

@author: Kitsune
"""

import pandas as pd
import numpy as np
import os
from sklearn import datasets

from sklearn.model_selection import train_test_split

form = pd.ExcelFile('kittest.xlsx')
#test = pd.ExcelFile('Example1_test.xlsx')

X_tr = form.parse('Features')
y_tr = form.parse('Target')
y=y_tr['Label']
latitude = [-35.3,
-15.75,
-6.17,
-6.17,
-1.26,
9.03,
11.55,
12.65,
13.75,
14.66,
14.91,
17.25,
17.98,
19.75,
23.76,
28.61,
30.03,
33.66,
34.03,
35.68,
35.7,
36.7,
38,
39.91,
39.91,
41.26,
41.33,
41.71,
41.9,
42.86,
44.41,
52.5,
54.68,
]
longitude = [149.12,
-47.95,
35.74,
106.82,
36.8,
38.74,
104.91,
-8,
100.48,
-17.41,
-23.51,
-88.76,
-76.8,
96.1,
121,
77.2,
31.21,
73.16,
-6.85,
51.41,
139.71,
3.21,
23.71,
32.83,
116.38,
69.21,
19.8,
44.78,
12.48,
74.6,
26.1,
-0.12,
25.31,
]
# =============================================================================
# from sklearn.naive_bayes import GaussianNB
# 
# clf = GaussianNB().fit(X_tr, y_tr)
# 
# y_pre = clf.predict(X_te)
# 
# from sklearn.metrics import confusion_matrix
# 
# print('X_te',X_te)
# print('accuracy',clf.score(X_te,y_te))
# print('conf_matrix',confusion_matrix(y_te, y_pre))
# print('probs',clf.predict_proba(X_te))
# =============================================================================

from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(X_tr)
print(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
             #, columns = ['pc1', 'pc2', 'pc3'])


print('lol')
print(pca.explained_variance_ratio_)

from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
#df = pd.read_csv('Absenteeism_at_work.csv', sep=';')
# =============================================================================
# maximum=1000000
# for ncluster in range(2,31):
#     labels = KMeans(n_clusters=ncluster, n_init=50, random_state=1234, tol=1e-10).fit_predict(X_tr)
#     avg = silhouette_score(X_tr,labels)
#     if maximum<avg:
#         maximum=avg
#         maxk=ncluster
#     print("For k=%d, the score is %g"%(ncluster,avg))
#     
# k=maxk
# =============================================================================
kitselect=2
km = KMeans(n_clusters=kitselect, n_init=50, random_state=1234, tol=1e-10).fit(X_tr)
labels = km.predict(X_tr)
np.savetxt("labels.csv", labels, delimiter=",")
colors = cm.tab20b(np.array(labels).astype(float)/kitselect)
#cm.Paired()

print(np.array(y).astype(float))

fig,ax = plt.subplots()
fig.set_size_inches(16.0, 10.38)
plt.ylim(-90, 90)
plt.xlim(-180,180)
ax.scatter(y_tr['longitude'],y_tr['latitude'],marker='o',s=100,c=colors)
i=0
#for kit,tin in zip(longitude,latitude):
#    i=i+1
#    ax.scatter(kit,tin,s=50,marker='$%d$'%labels,c='white')

# =============================================================================
# fig = plt.figure()
# fig.set_size_inches(18.5, 10.5)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(principalDf['pc1'],principalDf['pc2'],principalDf['pc3'],marker='o',s=30,c=colors)
# =============================================================================

#colors1= cm.coolwarm(np.array(solved).astype(float)/2)
#ax.scatter(x_new[:,0],x_new[:,1],marker='x',s=40,c=colors1)

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1234)