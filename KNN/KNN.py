# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:50:16 2016

@author: voroujak
"""

import numpy as np
import sklearn
from matplotlib import pyplot as plt
import matplotlib as mpl
import math
from math import exp
from sklearn.cross_validation import train_test_split 

from sklearn.grid_search import GridSearchCV
from sklearn import neighbors, datasets
from sklearn import decomposition
#from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
X=iris.data#[:,:2]
y=iris.target#[:,:2]

#standardize data
#X= sklearn.preprocessing.normalize(X)

X=sklearn.preprocessing.scale(X)


#PCA
pca=decomposition.PCA(n_components=4)
XxX=pca.fit_transform(X)
#KNN

X=[XxX[:,1],XxX[:,2]]
#X=[X[:,0],X[:,1]]
#X=pca.transform(X)
X=np.vstack(X)
X=X.T


X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.4)


alpha=1

def gauss(x):
    return np.exp(-alpha*np.power(x,2))# for x in [dis]
   # return guass

#n_neighbors=10
#for weights in ['uniform','distance',gauss]:
tuned_parameters=[{'n_neighbors':[1,2,3,4,5,6,7,8,9,10], 'weights':['uniform', 'distance', gauss]}]
neigh=GridSearchCV(neighbors.KNeighborsClassifier(),param_grid=tuned_parameters)
neigh.fit(X_train,Y_train)
neigh.score(X_test,Y_test)
print("score on test set is: "+str(neigh.score(X_test,Y_test)))
#plot Decision boundary

print(neigh.best_score_)
print(" estimation for n_neighbors:")
print(neigh.best_estimator_.n_neighbors)
print("estimation for weights:")
print(neigh.best_estimator_.weights)

h=.02
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min, x_max,h),
                  np.arange(y_min,y_max,h))
Z=neigh.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.figure()
cmap_light=mpl.colors.ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)

#plot the datasets
cmap_bold = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#plt.title("3-Class classification (k = %i, weights = '%s')"
#              % (n_neighbors))

plt.title("Best solution is obtained by K="+str(neigh.best_estimator_.n_neighbors)+ " and wight is: "+ str(neigh.best_estimator_.weights)+ "\n which on test sets gives score of: " + str(neigh.score(X_test,Y_test)))

plt.show()

