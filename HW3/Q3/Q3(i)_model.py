from __future__ import print_function
from __future__ import division
import pandas as pd
import pylab as pl
import numpy as np
from six import string_types
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
#from sklearn.utils import shuffle
from CV import KFold
from CV import train_test_split
import csv
import random
import matplotlib.pyplot as plt
import sys
from CV import shuffle_1

#choose algorithm
algorithm_set = ['LR','SVM']

#parameters for LR
sigma_sq = 0.01
reg_type = 'l2'

#classification threshold
classify_thres = np.arange(0.1,0.9,0.1)
#k-fold cross validation
k=20

#read training and validation data
df = pd.read_csv('movies_training.csv')
#storing accuracy
LR_accuracy_validation = np.zeros((len(classify_thres),k))
LR_accuracy_train = np.zeros((len(classify_thres),k))
#cross validation
threshold_idx = 0

for threshold in classify_thres:
    print(threshold)
    shuffled_data = shuffle_1(df)
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
        y_pred_validation = (model.predict_proba(data_validation)[:,1] > threshold) + 0
        f1score_validation = f1_score(target_validation, y_pred_validation)   
        
        
        #storing F1 score 
        LR_accuracy_validation[threshold_idx][k_fold_idx] = f1score_validation
            
        k_fold_idx += 1
    threshold_idx += 1

#plt.axes().set_xscale('log')
plt.plot(classify_thres,np.average(LR_accuracy_validation,axis=1),'r',label='LR')
plt.title('F1 score vs. threshold')
plt.xlabel('threshold')
plt.ylabel('F1 score')
plt.savefig("Q3(i).png")