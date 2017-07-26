from __future__ import print_function
from __future__ import division
import pandas as pd
import pylab as pl
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
#from sklearn.utils import shuffle
import csv
import random
import matplotlib.pyplot as plt
import sys
#from scipy import stats
from CV import KFold
from CV import shuffle
from CV import one_tailed_ttest

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def float_or_str(x):
	if isfloat(x):
		return (x)
	else:
		return (-1)


def percent_to_float(x):
	if isfloat(x):
		return (x/100)
	else:
		return float(x.strip('%'))/100

def add_noise(x):
	if not isinstance(x, string_types):
		return (x + np.random.normal(loc=0.0, scale=1e-3))
	else:
		return (x)


def data_processing(filename):
    # Read file (must be in UFT-8 if using python version >= 3)
    df = pd.read_csv(filename)

    # print (df.head()) # check feature ids

    df['Interest Rate Percentage'] = [percent_to_float(i) for i in df['Interest Rate Percentage']]

    df['Debt-To-Income Ratio'] = [percent_to_float(i) for i in df['Debt-To-Income Ratio Percentage']]

    features_to_keep = ['Amount Requested','Interest Rate Percentage','Loan Purpose','Loan Length in Months',
                        'Monthly PAYMENT','Total Amount Funded','FICO Range','Debt-To-Income Ratio Percentage']

    # convert interger values to float (helps avoiding optimization implementation issues)
    for feature in features_to_keep:
        if feature not in ['FICO Range','Loan Purpose']:
            df[feature] = [float(i) for i in df[feature]]

    # Scale values
    df['Total Amount Funded'] /= max(df['Total Amount Funded'])
    df['Amount Requested'] /= max(df['Amount Requested'])
    df['Loan Length in Months'] /= max(df['Loan Length in Months'])
    df['Monthly PAYMENT'] /= max(df['Monthly PAYMENT'])

    # Interaction terms
    df['Total Amount Funded * Requested'] = df['Total Amount Funded']*df['Amount Requested']
    df['Total Amount Funded * Requested'] /= max(df['Total Amount Funded * Requested'])

    df['Interest Rate Percentage * Monthly PAYMENT'] = df['Interest Rate Percentage']*df['Monthly PAYMENT']
    df['Interest Rate Percentage * Monthly PAYMENT'] /= max(df['Interest Rate Percentage * Monthly PAYMENT'])


    target_var = [float_or_str(i) for i in df['Status']]

    # create a clean data frame for the regression
    data = df[features_to_keep].copy()
    
    data['intercept'] = 1.0

    return (data,target_var)

def add_categorical(train, feature_str):
    # encode categorical features
    encoded = pd.get_dummies(train[feature_str])#, dummy_na=True)
    train_rows = train.shape[0]
    train_encoded = encoded.iloc[:train_rows, :]
    validation_encoded = encoded.iloc[train_rows:, :] 

    train_encoded_wnoise = train_encoded.applymap(add_noise)
    #validation_encoded_wnoise = validation_encoded.applymap(add_noise)

    train.drop(feature_str,axis=1, inplace=True)
    #validation.drop(feature_str,axis=1, inplace=True)

    train = train.join(train_encoded_wnoise.ix[:, :])
    #validation = validation.join(validation_encoded_wnoise.ix[:, :])

    return train


train_file = "Bank_Data_SeparateTrain.csv"
validation_file = "Bank_Data_SeparateValidation.csv"

# L2 Regularization penalty \\sigma^2/\\sigma^2_\\beta = sigma_sq
sigma_sq = 1e-2
reg_type = "l2"
reg_var = "1/\\sigma^2_w"

F1_scores_validation = []
F1_scores_train = []

#[kira]: shuffle data
data, target = data_processing("Bank_Data_Train.csv")
data = add_categorical(train=data,feature_str='FICO Range')
data = add_categorical(train=data,feature_str='Loan Purpose')

#k-fold
k = 20
#classification threshold for LR
LR_threshold = 0.35
#record accuracy score
LR_accuracy_validation = np.zeros(k)
LR_accuracy_train = np.zeros(k)
SVC_accuracy_validation = np.zeros(k)
SVC_accuracy_train = np.zeros(k)
algorithm_set = ['LR','SVC']
array_idx = 0

#to save time
offline = False
#t-test threshold
confidence = 0.05
#penalty of svm
penalty_svm = 40

#[kira]:for each training size, get F1 score
#if we have data recorded
if offline == False:
    for algorithm in algorithm_set:
        #always shuffle data before algorithm
        shuffled_data, shuffled_target = shuffle(data,target)
        
        indice = KFold(data, k)
        k_fold_idx = 0
        for kfold_idx in range(0,k): 
            #[kira]:divide data
            data_validation = shuffled_data.iloc[indice[kfold_idx]:indice[kfold_idx+1],:]
            target_validation = shuffled_target[indice[kfold_idx]:indice[kfold_idx+1]]
            if k_fold_idx == 0:
                data_train = shuffled_data.iloc[indice[kfold_idx+1]:,:]
                target_train = shuffled_target[indice[kfold_idx+1]:]
            elif k_fold_idx == k-1:
                data_train = shuffled_data.iloc[0:indice[kfold_idx],:]
                target_train = shuffled_target[0:indice[kfold_idx]]
            else:
                data_train = pd.concat([ shuffled_data.iloc[0:indice[kfold_idx],:] , shuffled_data.iloc[indice[kfold_idx+1]:,:] ])
                target_train = np.concatenate((shuffled_target[0:indice[kfold_idx]] , shuffled_target[indice[kfold_idx+1]:]))
            
            # Describe classifier and regularization type
            if algorithm == 'LR':
                model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
            else:
                model = svm.SVC(C=penalty_svm,probability=False)
            # Train model
            model.fit(X=data_train, y=target_train)
            # Predicted probabilities of label +1
            #     0.5 is an arbitrary number
            if algorithm == 'LR':
                y_pred_validation = (model.predict_proba(data_validation)[:,1] > LR_threshold) + 0
                y_pred_train = (model.predict_proba(data_train)[:,1] > LR_threshold) + 0
            else:
                y_pred_validation = model.predict(data_validation)
                y_pred_train = model.predict(data_train)
            
            f1score_validation = f1_score(target_validation, y_pred_validation)        
            f1score_train = f1_score(target_train, y_pred_train)
            
            #get accuracy of training and validation set
            if algorithm == 'LR':
                LR_accuracy_train[k_fold_idx] = f1score_train
                LR_accuracy_validation[k_fold_idx] = f1score_validation
            else:
                SVC_accuracy_train[k_fold_idx] = f1score_train
                SVC_accuracy_validation[k_fold_idx] = f1score_validation
            
            k_fold_idx += 1
    
    #output each F1 score
    print(LR_accuracy_validation)
    print(SVC_accuracy_validation)
    with open('Q2(v)_LR.txt','w') as f:
        for i in LR_accuracy_validation:
            f.write('%f,'%i)
    with open('Q2(v)_SVC.txt','w') as f:
        for i in SVC_accuracy_validation:
            f.write('%f,'%i)
    
    #paired t-Test
    (t_score,p_value) = one_tailed_ttest(LR_accuracy_validation, SVC_accuracy_validation)
    print(t_score)
    print(p_value)
    if p_value < confidence:
        if t_score > 0:
            print("LR is more accurate")
        else:
            print("SVM is more accurate")
    else:
        print("we cannot differentiate the power of LR and SVC")
else:
    with open('Q2(v)_LR.txt','r') as f:
        line = f.readline()
        LR_accuracy_validation = line.rstrip(',').split(',')
        LR_accuracy_validation = map(float,LR_accuracy_validation)
        print(LR_accuracy_validation)
    with open('Q2(v)_SVC.txt','r') as f:
        line = f.readline()
        SVC_accuracy_validation = line.rstrip(',').split(',')
        SVC_accuracy_validation = map(float,SVC_accuracy_validation)
        print(SVC_accuracy_validation)
        
    #paired t-Test
    (t_score,p_value) = one_tailed_ttest(LR_accuracy_validation, SVC_accuracy_validation)
    print(t_score)
    print(p_value)
    if p_value < confidence:
        if t_score > 0:
            print("LR is more accurate")
        else:
            print("SVM is more accurate")
    else:
        print("we cannot differentiate the power of LR and SVC")