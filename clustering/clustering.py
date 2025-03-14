# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 20:57:30 2016

@author: voroujak
"""

from sklearn import datasets
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib as mpl
from sklearn.mixture import GMM
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import homogeneity_score

digits=datasets.load_digits()
X=digits.data
Y=digits.target
#standardize and applying PCA

x=X[Y<5]
y=Y[Y<5]
X=sklearn.preprocessing.scale(x)
X_t=PCA(n_components=2).fit_transform(x)

x0=X_t[y==0]
x1=X_t[y==1]
x2=X_t[y==2]
x3=X_t[y==3]
x4=X_t[y==4]
#x1=X[Y==1]



#now trying to using KMeans
kmeans=KMeans(n_clusters=10)
kmeans.fit(X_t)

h=.02
x_min, x_max = X_t[:,0].min()-1, X_t[:,0].max()+1
y_min, y_max = X_t[:,1].min()-1, X_t[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min, x_max,h),
                  np.arange(y_min,y_max,h))
Z=kmeans.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.figure()
plt.clf()
plt.imshow(Z,interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
#plt.plot(X_t[:,0],X_t[:,1],'k.',markersize=8)
plt.scatter(x0[:,0],x0[:,1],c='r')
plt.scatter(x1[:,0],x1[:,1],c='b')
plt.scatter(x2[:,0],x2[:,1],c='g')
plt.scatter(x3[:,0],x3[:,1],c='y')
plt.scatter(x4[:,0],x4[:,1],c='w')

plt.legend(["0", "1" ,"2" ,"3" ,"4" ])

centroids=kmeans.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],
            marker='x', s=169, linewidth=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

"""""""""" now trying to use GMM clustering"""""""""""



def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes
    
    :param clusters: the cluster assignments array
    :type clusters: numpy.array
    
    :param classes: the ground truth classes
    :type classes: numpy.array
    
    :returns: the purity score
    :rtype: float
    """
    
    A = np.c_[(clusters,classes)]

    n_accurate = 0.

    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]


homm =list()
nm = list()
pr=list()

for K in range(2,11):
    GM=GMM(n_components=K)
    GM.fit(X_t)

    zozo=GM.predict(X_t)
    NMI= normalized_mutual_info_score(GM.predict(X_t),y)
    HoS=homogeneity_score(y,GM.predict(X_t))
    purr=purity_score(GM.predict(X_t),y)
    #print("The NMI is : "+ str(NMI))
    #print("then homogenity is : " + str(HoS))
    #print("then Purity is: "+ str(purity_score(GM.predict(X_t),y)))
    homm.append(HoS)
    nm.append(NMI)
    pr.append(purr)
xax=[2,3,4,5,6,7,8,9,10]
plt.plot(xax,homm, c='r')
plt.plot(xax,nm)
plt.plot(xax,pr)
plt.legend(["homogenity", "NMI", "purity"], loc="lower center")

plt.show()
    
    

