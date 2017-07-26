from __future__ import print_function
from __future__ import division
import pandas as pd
import pylab as pl
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.utils import shuffle
from CV import KFold
from CV import train_test_split
import csv
import random
import matplotlib.pyplot as plt
import sys

#parameters for LR
sigma_sq = 0.001
reg_type = 'l2'

#k-fold cross validation
k=20
#read training and validation data
df = pd.read_csv('movies_training_pearson.csv')
#storing accuracy
LR_accuracy_validation = np.zeros(k)
LR_accuracy_train = np.zeros(k)

classify_thres = 0.4
shuffled_data = shuffle(df)
indice = KFold(df, k)
k_fold_idx = 0
for kfold_idx in range(0,k): 
    #[kira]:divide data
    data_validation = shuffled_data.iloc[indice[kfold_idx]:indice[kfold_idx+1],:]
    if k_fold_idx == 0:
        data_train = shuffled_data.iloc[indice[kfold_idx+1]:,:]
    elif k_fold_idx == k-1:
        data_train = shuffled_data.iloc[0:indice[kfold_idx],:]
    else:
        data_train = pd.concat([ shuffled_data[0:indice[kfold_idx]] , shuffled_data[indice[kfold_idx+1]:] ])
    target_train = data_train['target_movie_Pulp_Fiction_1994']
    data_train = data_train.iloc[:,1:(data_train.shape[1]-1)]
    target_validation = data_validation['target_movie_Pulp_Fiction_1994']
    data_validation = data_validation.iloc[:,1:(data_validation.shape[1]-1)]

    #build Model
    model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)

    #fit model    
    model.fit(X=data_train, y=target_train)
    
    y_pred_validation = (model.predict_proba(data_validation)[:,1] > classify_thres) + 0
    
    #accuracy
    accuracy = 0
    copy_target = np.array(target_validation)
    for i in range(0,len(y_pred_validation)):
        if y_pred_validation[i] == copy_target[i]:
            accuracy += 1
    accuracy = accuracy /float(len(y_pred_validation))
    print(accuracy)
            
    #record accuracy
    LR_accuracy_validation[k_fold_idx] = accuracy
        
    k_fold_idx += 1
    
print(LR_accuracy_validation)