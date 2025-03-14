# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:45:51 2016

@author: voroujak
"""
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import preprocessing
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


XTrain= np.load("regressionData/regression_Xtrain.npy")
XTest= np.load("regressionData/regression_Xtest.npy")
YTrain= np.load("regressionData/regression_ytrain.npy")
YTest= np.load("regressionData/regression_ytest.npy")
lr=LinearRegression()
mmse= []
for K in range( 1,  10):

    poly=PolynomialFeatures(degree=K, include_bias=False)
    xPoly=poly.fit_transform(XTrain.reshape(-1,1))
    lr=LinearRegression()
    lr.fit(xPoly,YTrain)
    x_range=np.linspace(-1,5.5,50)
    predicted1=lr.predict(poly.fit_transform(x_range.reshape(-1,1)))
    predicted2=lr.predict(poly.fit_transform(XTest.reshape(-1,1)))

    MSE=mean_squared_error(predicted2,YTest)
    print (MSE)
    mmse.append(MSE)
    plt.plot(x_range.reshape(-1,1),predicted1)
    plt.scatter(XTest.reshape(-1,1),YTest,c='r')
    plt.scatter(XTrain.reshape(-1,1),YTrain, c='b')
    plt.title("Degree of " + str (K) + " with MSE of: "+ str(MSE))
    plt.show()
plt.title("Mean square error of various degrees")
plt.xlabel("Degrees")
plt.ylabel("MeanSquareError")

plt.plot([1,2,3,4,5,6,7,8,9],mmse)





