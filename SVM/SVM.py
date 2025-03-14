# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:09:25 2016

@author: voroujak
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split 

from sklearn.grid_search import GridSearchCV


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .02  # step size in the mesh

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.3)
X_Train, X_Test, Y_Train, Y_Test= X_train, X_test, Y_train, Y_test
X_train, X_valid, Y_train, Y_valid=train_test_split(X_train,Y_train,test_size=0.4)


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
ci=[  0.01,  0.1 , 1 ,10 ,100 , 1000,10000]  # SVM regularization parameter
scores=list()
for C in ci:
    svc = svm.SVC(kernel='rbf', C=C).fit(X_train, Y_train)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape) 
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("With C="+ str(C)+ " the obtained score on validation is: "+ str(svc.score(X_valid,Y_valid)))
    plt.show()
    scores.append(svc.fit(X_train,Y_train).score(X_valid,Y_valid))
    
print (scores)
plt.plot(ci,scores)
plt.xscale('log')
plt.title("Scores of different C on validation set")
plt.show()
print(scores.index(max(scores)))
C=ci[scores.index(max(scores))]
print (C)
#trying to classify test datas
svc = svm.SVC(kernel='rbf', C=C).fit(X_train, Y_train)
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) 
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=plt.cm.coolwarm)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("test datas on desicion boundries scored " + str(svc.score(X_test,Y_test)))
plt.show()
print("9.The result on test set is:")
print(svc.score(X_test,Y_test))


tuned_parameters=[{'kernel':['rbf'],'gamma':[10**-9, 10**-7, 10**-5, 10**-3, 10**-1,1], 'C':[0.01, 0.1, 1, 10, 100,1000,10000]}]
svc_grid=GridSearchCV(svm.SVC(),param_grid=tuned_parameters)
svc_grid.fit(X_train,Y_train)
svc_grid.score(X_test,Y_test)
print("11.tuned by grid on validation")
print(svc_grid.score(X_valid,Y_valid))

print("13. evaluate best parameters on test set")
print(svc_grid.score(X_test,Y_test))
i=0
j=0

for gamma in [10**-9, 10**-7, 10**-5, 10**-3, 10**-1,1]:
    j=0
    for C in [0.01, 0.1, 1, 10, 100,1000,10000]:
        svc=svm.SVC(kernel='rbf', gamma=gamma, C=C)
        svc.fit(X_train,Y_train)
        print("gamma="+str(i)+"c="+ str(j))
        print(svc.score(X_valid, Y_valid))
        j+=1
    i+=1

print("now K-fold cross validation")
 

       
gamma_vals=[10**-9, 10**-7, 10**-5, 10**-3, 10**-1, 1]
C_vals=[0.01, 0.1, 1, 10, 100,1000,10000]
k_folds=[0,1,2,3,4]
X_folds=np.array_split(X_Train,5)
Y_folds=np.array_split(Y_Train,5)      
scores=np.empty((len(gamma_vals),len(C_vals),len(k_folds)))
for i, gamma in enumerate(gamma_vals):
    for j, C in enumerate (C_vals):
        for k, fold in enumerate(k_folds):
            X_Training=list(X_folds)
            X_testing=X_Training.pop(k)
            X_Training= np.concatenate(X_Training)
            Y_training=list(Y_folds)
            Y_testing=Y_training.pop(k)
            Y_training=np.concatenate(Y_training)
            svc=svm.SVC(kernel='rbf', gamma=gamma, C=C)

            scores[i,j,k]=(svc.fit(X_Training,Y_training).score(X_testing,Y_testing))
            #scores[i,j,k]= model.score(fold.validation_data, fold.validation_labels)
        
MAT = np.sum(scores,axis=2)/5
print(MAT)
print(np.argmax(MAT))
gammaindex=int((np.argmax(MAT)-1)/7 )
Cindex=np.argmax(MAT)%7 
scv=svm.SVC(kernel='rbf', gamma=gamma_vals[gammaindex], C=C_vals[Cindex])
print("gamma index is: ")
print(gammaindex)
print("C is: ")
print(Cindex)

print("the result with 5fold validation on test set is:")
print(scv.fit(X_Train,Y_Train).score(X_test,Y_test))


 
