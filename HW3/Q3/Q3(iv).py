from __future__ import print_function
from __future__ import division
import pandas as pd
import pylab as pl
import numpy as np
from six import string_types
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
#from sklearn.utils import shuffle
from CV import KFold
from CV import train_test_split
from CV import shuffle_1
import csv
import random
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
from CV import one_tailed_ttest

#choose algorithm
algorithm_set = ['logistic','linear']

#parameters for LR
sigma_sq = 0.01
reg_type = 'l2'

#classification threshold
threshold = 0.4
#k-fold cross validation
k=20
#p-value threshold
p_value_thres = 0.05

#read training and validation data
df = pd.read_csv('movies_training_cosine.csv')
#storing accuracy
logistic_accuracy_validation = np.zeros(k)
logistic_accuracy_train = np.zeros(k)
linear_accuracy_validation = np.zeros(k)
linear_accuracy_train = np.zeros(k)

for algorithm in algorithm_set:
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
        if algorithm =='logistic':
            model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
        elif algorithm == 'linear':
            model = SGDClassifier(loss='squared_loss',penalty='none',fit_intercept=True,shuffle=False, \
                                  random_state=0,eta0=0.01,power_t=0.5)
        #fit model    
        model.fit(X=data_train, y=target_train)
        #prediction on validation set
        if algorithm =='logistic':
            y_pred_validation = (model.predict_proba(data_validation)[:,1] > threshold) + 0
        elif algorithm == 'linear':
            y_pred_validation = model.predict(data_validation)
        #f1 score
        f1score_validation = f1_score(target_validation, y_pred_validation)   
        
        #storing F1 score 
        if algorithm == 'logistic':
            logistic_accuracy_validation[k_fold_idx] = f1score_validation
        elif algorithm == 'linear':
            linear_accuracy_validation[k_fold_idx] = f1score_validation 
            
        k_fold_idx += 1



#plot distribution
data_logistic = sorted(logistic_accuracy_validation)
fit_logistic = stats.norm.pdf(data_logistic, np.mean(data_logistic), np.std(data_logistic))
pl.plot(data_logistic,fit_logistic,'-o',label='Logistic Regression')

data_linear = sorted(linear_accuracy_validation)
fit_linear = stats.norm.pdf(data_linear, np.mean(data_linear), np.std(data_linear))
pl.plot(data_linear,fit_linear,'-v',label='SGDClassifier')

plt.title('F1 score vs. threshold')
plt.xlabel('F1 score')
plt.ylabel('Frequency')
plt.legend(loc="upper left")
plt.savefig("Q3(iv).png")

(t_score,p_value) = one_tailed_ttest(logistic_accuracy_validation, linear_accuracy_validation)
print(t_score)
print(p_value)
if p_value < p_value_thres:
    print("Logistic Regression is more powerful")
else:
    print("We cannot differentiate the power of the two")