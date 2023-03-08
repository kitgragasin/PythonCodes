# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:12:34 2019

@author: Kitsune
"""

import pandas as pd
import numpy as np
import os
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

form = pd.ExcelFile('kittest.xlsx')
#test = pd.ExcelFile('Example1_test.xlsx')

X_tr = form.parse('Features')
print(X_tr)
selector= VarianceThreshold()
X_tr = selector.fit_transform(X_tr)
print(X_tr)
y_tr = form.parse('Target')
y=y_tr['Label']


X_train, X_test, y_train, y_test = train_test_split(X_tr,y,test_size=0.3, random_state=1234)
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
neighbors=5
print("This is for KNN %d"%neighbors)
knn = KNeighborsClassifier(n_neighbors=neighbors).fit(X_train,y_train)
solved=knn.predict(X_test)
#print(solved)
print("KNN test difference")
print(np.mean(solved==y_test))
print("KNN test score")
print(knn.score(X_test,y_test))
print ("Zip:")
average_dist = 0.0
ctr=0.0
for x, y in zip(y_test, solved):
    average_dist=average_dist + haversine(longitude[x-1], latitude[x-1], longitude[y-1], latitude[y-1])
    ctr = ctr+1
    
average_dist = average_dist/ctr
print(average_dist)
from sklearn.model_selection import cross_val_score
print("cross val score")
scores = cross_val_score(knn, X_train, y_train, cv=5)

print(scores)
# =============================================================================
from sklearn.naive_bayes import GaussianNB
 
clf = GaussianNB().fit(X_train, y_train)
y_pre = clf.predict(X_test)
 
from sklearn.metrics import confusion_matrix
print("This is for GaussianNaiveBayes")
 #print('X_te',X_test)
print('accuracy',clf.score(X_test,y_test))
print('conf_matrix',confusion_matrix(y_test, y_pre))
#print('probs',clf.predict_proba(X_test))
solved=y_pre
average_dist = 0.0
ctr=0.0
for x, y in zip(y_test, solved):
    average_dist=average_dist + haversine(longitude[x-1], latitude[x-1], longitude[y-1], latitude[y-1])
    ctr = ctr+1
    
average_dist = average_dist/ctr
print(average_dist)
 
# =============================================================================
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=20)
neigh.fit(X_train)
#print("Try ang nearest neighbors")
#print(X_test)
newneighbors = neigh.kneighbors(X_test, return_distance = False)
print(newneighbors)
#print("bago mag forloop")
neighborlabels=[]
solved=[]
for row,testime in zip(newneighbors,X_test):

    rowlabels=[]
#    print(X_train[row[0],:])
    testsample=np.array([X_train[row[0],:],X_train[row[1],:],X_train[row[2],:],X_train[row[3],:],X_train[row[4],:]])
#    print(testsample)
    fsample=list(testsample)
    sample=[X_train[row[0],:],X_train[row[1],:],X_train[row[2],:].reshape(1,-1),X_train[row[3],:],X_train[row[4],:]]
#    sample=[sample]
 #   print("Sample is")
  #  print(sample)
    label=[y_train.values[row[0]],y_train.values[row[1]],y_train.values[row[2]],y_train.values[row[3]],y_train.values[row[4]]]
#    print("label is")
 #   print(label)
    clf = GaussianNB().fit(fsample, label)
  #  print("naka naive bayes")
   # print(testime)
    kit_pre = clf.predict([testime])
    #print(kit_pre)
    solved.append(kit_pre[0])

#print(solved)

#print(y_test)
print("This is for Naive Bayes for 5 Nearest Neighbors")
print("accuracy",np.mean(solved==y_test))
for x, y in zip(y_test, solved):
    average_dist=average_dist + haversine(longitude[x-1], latitude[x-1], longitude[y-1], latitude[y-1])
    ctr = ctr+1
    
average_dist = average_dist/ctr
print("average distance error",average_dist)
    
# =============================================================================
#     for item in row:
#         print("item")
#         print(item)
#         print(X_train[item,:])
#         print(y_train[item,:,0])
#         rowlabels=np.append(rowlabels,y_train.values[item])
#     
#     print(rowlabels)
#     print(y_train)
#     neighborlabels=np.concatenate(neighborlabels,rowlabels, axis=0)
# =============================================================================



# =============================================================================
# fig,ax = plt.subplots()
# ax.scatter(principalDf['pc1'],principalDf['pc2'],marker='o',s=30,c=colors)
# =============================================================================
# =============================================================================
# fig = plt.figure()
# fig.set_size_inches(18.5, 10.5)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(principalDf['pc1'],principalDf['pc2'],principalDf['pc3'],marker='o',s=30,c=colors)
# =============================================================================

#colors1= cm.coolwarm(np.array(solved).astype(float)/2)
#ax.scatter(x_new[:,0],x_new[:,1],marker='x',s=40,c=colors1)


