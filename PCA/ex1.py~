# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 21:52:40 2016

@author: voroujak
"""

from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import preprocessing
import os
from sklearn.cross_validation import train_test_split 
from sklearn.naive_bayes import GaussianNB

os.chdir("Smpls")
filelist=os.listdir()

data_list=[]

for file in filelist:
    img_data=np.asarray(Image.open(file))
    x=np.asarray(img_data.ravel())
    data_list.append(x)
    
    
"""
Making y, as output of classifier
"""    
y0=[0]*72
y1=[1]*72
y2=[2]*72
y3=[3]*72
y=np.concatenate((y0,y1,y2,y3))

"""
ending of making y
"""
    
X=np.array(data_list)
X_norm= preprocessing.normalize(X)
X_scale=preprocessing.scale(X_norm)
#X_scale=preprocessing.scale(X)

X_t=PCA(n_components=20).fit_transform(X_scale)
#choosing which component to take part in GaussianNB
XX_tt=[X_t[:,2],X_t[:,3]]
x_tT=np.vstack(XX_tt)
x_tT=x_tT.T

xx=2
yy=3
plt.scatter(X_t[0:71,xx],X_t[0:71,yy],c='y')    
plt.scatter(X_t[72:143,xx],X_t[72:143,yy],c='m') 
plt.scatter(X_t[144:215,xx],X_t[144:215,yy],c='r') 
plt.scatter(X_t[216:287,xx],X_t[216:287,yy],c='g') 
plt.title("tenth and eleventh components")

X_train, X_test, y_train, y_test=train_test_split(x_tT,y,test_size=0.3)
gnb=GaussianNB()
XXXXXX=gnb.fit(X_train,y_train)
#print (gnb.predict(X_test))
print ("the score result on test is:")
print (gnb.score(X_test,y_test))
