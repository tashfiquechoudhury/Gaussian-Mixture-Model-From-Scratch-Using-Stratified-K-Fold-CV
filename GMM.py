# -*- coding: utf-8 -*-
"""
File:   GMM.py
Author:   Tashfique Hasnine Choudhury  
Date:   10.25.2021
Desc:   Implementation of GMM on iris data
    
"""

import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from itertools import permutations

"""Loading Iris data"""
data_iris =  load_iris()
A = data_iris.data
Y = data_iris.target

"""Splitting into train, test and validation data Initial approach"""
X_train, X_test, y_train, y_test = train_test_split(A, Y, test_size=0.2, random_state=1) # Train 120, Test 30
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # Train 90, Validation 30

def EM_GaussianMixture(X, NumberOfComponents, cov):
    
    MaximumNumberOfIterations = 150
    DiffThresh = 1e-5
    N = X.shape[0] #number of data points
    d = X.shape[1] #dimensionality
    rp = np.random.permutation(N) #random permutation of numbers 1:N
 
    """For best model (initial approach) rp = [16 89 55 33 38 64 58 63 26 17 67 15 62 80 12 65 14  3 22 23 20 28 69 34
    81 59  6 21  8 83 76 35  9 24 30  2 39 10 41 68 47 57 42 84 82 56 66 78
    44 51 19 25  1 36 49 32  0  7 79 70 45 31 61 87  5 54 27 43 52 75 46 18
    71 74 29 48 86 85  4 77 40 11 13 53 60 73 88 50 72 37]"""
     
    #Initialize Parameters
    Means = X[rp[0:NumberOfComponents],:]
    Sigs = np.zeros((d,d,NumberOfComponents))
    Ps = np.zeros((NumberOfComponents,))
    pZ_X = np.zeros((N,NumberOfComponents))

    for i in range(NumberOfComponents):
        if cov == 'Diagonal':
            Sigs[:,:,i] = np.eye(d) + np.array([[1,0,0,0],[0,3,0,0],[0,0,5,0],[0,0,0,7]])
        elif cov == 'Isotropic':
            Sigs[:,:,i] = np.eye(d)
        elif cov == 'Full':
            Sigs[:,:,i] = np.full((d,d),1)
        else:
            sys.exit('Please type Diagonal/Isotropic/Full')
            
        Ps[i] = 1/NumberOfComponents

    #Solve for p(z | x, Theta(t))
    for k in range(NumberOfComponents):
        mvn = stats.multivariate_normal(Means[k,:],Sigs[:,:,k], allow_singular=True)
        pZ_X[:,k] = mvn.pdf(X)*Ps[k]

    pZ_X = pZ_X / pZ_X.sum(axis=1)[:,np.newaxis]  # np.newaxis fixes cannot broadcast (N,d) (N,) errors

    Diff = np.inf
    NumberIterations = 1
    
    while Diff > DiffThresh and NumberIterations <= MaximumNumberOfIterations:
        #Update Means, Sigs, Ps
        MeansOld = Means.copy()
        SigsOld = Sigs.copy()
        PsOld = Ps.copy()
        for k in range(NumberOfComponents):
            #Means
            Means[k,:] = np.sum(X*pZ_X[:,k,np.newaxis],axis=0)/pZ_X[:,k].sum()
            
            #Sigs
            xDiff = X - Means[k,:] 
            J = np.zeros((d,d))
            for i in range(N):
                J = J + pZ_X[i,k]*np.outer(xDiff[i,:],xDiff[i,:])
            Sigs[:,:,k] = J / pZ_X[:,k].sum()
            
            #Ps
            Ps[k] = pZ_X[:,k].sum()/N

        #Solve for p(z | x, Theta(t))
        for k in range(NumberOfComponents):
            mvn = stats.multivariate_normal(Means[k,:],Sigs[:,:,k], allow_singular=True)
            pZ_X[:,k] = mvn.pdf(X)*Ps[k]
        pZ_X = pZ_X / pZ_X.sum(axis=1)[:,np.newaxis]
    
        Diff = abs(MeansOld - Means).sum() + abs(SigsOld - Sigs).sum() + abs(PsOld - Ps).sum();
        NumberIterations = NumberIterations + 1
    return Means, Sigs, Ps, pZ_X

def Pred(X, Y, Means, Sigs, Ps): #Prediction function using obtained Means, Sigs and Ps from the model using training data
    N = X.shape[0]
    pZX = np.zeros((N,NumberOfComponents))
    for k in range(NumberOfComponents):
        mn = stats.multivariate_normal(Means[k],Sigs[:,:,k])
        pZX[:,k] = mn.pdf(X)*Ps[k]

    pZX = pZX / pZX.sum(axis=1)[:,np.newaxis]

    cluster =[]
    for prob in pZX:
        cluster.append(np.argmax(prob))
    cluster = np.array(cluster)
    # Mapping to true labels, source: https://stackoverflow.com/questions/11683785/how-can-i-match-up-cluster-labels-to-my-ground-truth-labels-in-matlab
    assert cluster.ndim == 1 == Y.ndim
    assert len(cluster) == len(Y)
    cluster_names = np.unique(cluster)
    accuracy = 0

    perms = np.array(list(permutations(np.unique(Y))))

    remapped_labels = Y
    for perm in perms:
        flipped_labels = np.zeros(len(Y))
        for label_index, label in enumerate(cluster_names):
            flipped_labels[cluster == label] = perm[label_index]

        testAcc = np.sum(flipped_labels == Y) / len(Y)
        if testAcc > accuracy:
            accuracy = testAcc
            remapped_labels = flipped_labels #The cluster formation process is of random order/index, so to match with the order/index of the true values we use this segment
    return remapped_labels

def CMnAC(y_t, y_p):
    print('Accuracy =',accuracy_score(y_t, y_p))
    print('Confusion Matrix =')
    print(confusion_matrix(y_t, y_p))
    return

"""Training the model"""
NumberOfComponents=3
Means, Sigs, Ps, pZ_X = EM_GaussianMixture(X_train, NumberOfComponents, cov = 'Full')
print('OBTAINING BEST MODEL USING MULTIPLE RUNS')
"""Visualizing Clusters formed for the train_data """
for i in range(NumberOfComponents):
    fig = plt.figure(figsize=(6,6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    x = X_train[:,0] #x axis as first dimension
    y = X_train[:,1] #y axis as second dimension
    z = X_train[:,2] #z axis as third dimension
    s = X_train[:,3] #shape as fourth dimension
    
    m1a=Means[0,0]
    m1b=Means[0,1]
    m1c=Means[0,2]
    m1d=Means[0,3]
    m2a=Means[1,0]
    m2b=Means[1,1]
    m2c=Means[1,2]
    m2d=Means[1,3]
    m3a=Means[2,0]
    m3b=Means[2,1]
    m3c=Means[2,2]
    m3d=Means[2,3]
    
    img = ax.scatter(x, y, z, s=s, c=pZ_X[:,i], cmap=plt.copper()) #color as the fifth dimension showing clusters
    img = ax.scatter(m1a, m1b, m1c, s=m1d, c='r')
    img = ax.scatter(m2a, m2b, m2c, s=m2d, c='r')
    img = ax.scatter(m3a, m3b, m3c, s=m3d, c='r')
    #plt.savefig(f'Learned Clusters {i}')
    plt.show()
    
"""Prediction on Validation, Test and Traindata"""    
y_pred_val = Pred(X_val, y_val, Means=Means, Sigs=Sigs, Ps=Ps)
y_pred_test = Pred(X_test, y_test, Means=Means, Sigs=Sigs, Ps=Ps)
y_pred_train = Pred(X_train, y_train, Means=Means, Sigs=Sigs, Ps=Ps)

"""Visualizing the the Accuracy and Confusion Matrix"""
print('\nAccuracy and CM for Validation data:')
CMnAC(y_val, y_pred_val)
print('\nAccuracy and CM for Test data:')
CMnAC(y_test, y_pred_test)
print('\nAccuracy and CM for Train data:')
CMnAC(y_train, y_pred_train)


""" 2nd Approach """
print('------------------------------------')
print('\n\nOBTAINING BEST MODEL FROM CROSS VALIDATION')


"""Splitting into hold out test set and stratified train and validation data selected approach"""
X_train, X_test, y_train, y_test = train_test_split(A, Y, test_size=0.2, random_state=1) #Hold out test set
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
NumberOfComponents=3
M=[]
S=[]
P=[]
Accuracy_Valid=[]
Accuracy_Train=[]
XTrain=[]
XVal=[]
YTrain=[]
YVal=[]
for train_index, test_index in skf.split(X_train, y_train):
#    print("TRAIN:", train_index, "VALID:", test_index)
#    print("Length Train", len(train_index), "Length Valid", len(test_index))
    X_Train, X_val = X_train[train_index], X_train[test_index]
    y_Train, y_val = y_train[train_index], y_train[test_index]
    Means, Sigs, Ps, pZ_X = EM_GaussianMixture(X_Train, NumberOfComponents, cov = 'Diagonal')
    XTrain.append(X_Train)
    XVal.append(X_val)
    YTrain.append(y_Train)
    YVal.append(y_val)
    M.append(Means)
    S.append(Sigs)
    P.append(Ps)
    # Calculation for all folds for Validation set
    y_pred = Pred(X_val, y_val, Means=Means, Sigs=Sigs, Ps=Ps)
    Accuracy = accuracy_score(y_val, y_pred)
    print('Validation Accuracy', Accuracy)
    print(confusion_matrix(y_val, y_pred))
    Accuracy_Valid.append(Accuracy)

    # Calculation for all folds for Train set
    y_predTr = Pred(X_Train, y_Train, Means=Means, Sigs=Sigs, Ps=Ps)
    AccuracyTr =accuracy_score(y_Train, y_predTr)
    print('Train Accuracy', AccuracyTr)
    print(confusion_matrix(y_Train, y_predTr))
    Accuracy_Train.append(AccuracyTr)
    print('------------------------------------')

MaxVal=max(Accuracy_Valid) 
maxid=Accuracy_Valid.index(MaxVal) #Getting index for the best split
Final_Mean=np.array(M[maxid]) #Getting best Mean
Final_Sigs=np.array(S[maxid]) #Getting best Sigs
Final_Ps=np.array(P[maxid]) #Getting best Ps
Final_XTrain=np.array(XTrain[maxid]) #Getting X Train Set
Final_XVal=np.array(XVal[maxid]) #Getting X Validation Set
Final_YTrain=np.array(YTrain[maxid]) #Getting Y Train Set
Final_YVal=np.array(YVal[maxid]) #Getting Y Validation Set

print('------------------------------------')
print('==============SELECTED MODEL==============')
y_pred_testCV = Pred(X_test, y_test, Means=Final_Mean, Sigs=Final_Sigs, Ps=Final_Ps)
y_pred_trainCV = Pred(Final_XTrain, Final_YTrain , Means=Final_Mean, Sigs=Final_Sigs, Ps=Final_Ps)
y_pred_valCV = Pred(Final_XVal, Final_YVal, Means=Final_Mean, Sigs=Final_Sigs, Ps=Final_Ps)
print('\nAccuracy and CM for Validation data:')
CMnAC(Final_YVal, y_pred_valCV)
print('\nAccuracy and CM for Test data:')
CMnAC(y_test, y_pred_testCV)
print('\nAccuracy and CM for Train data:')
CMnAC(Final_YTrain, y_pred_trainCV)
print('\nStandard deviation for Validation Accuracy', np.std(Accuracy_Valid))
print('\nStandard deviation for Train Accuracy',np.std(Accuracy_Train))
print('\nMean Train Accuracy', np.mean(Accuracy_Train))
print('\nMean Validation Accuracy', np.mean(Accuracy_Valid))