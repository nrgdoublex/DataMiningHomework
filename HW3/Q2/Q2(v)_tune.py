from __future__ import print_function
from __future__ import division
import pandas as pd
import pylab as pl
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
#from sklearn.utils import shuffle
from sklearn import svm
#from sklearn.model_selection import KFold
import csv
import random
import matplotlib.pyplot as plt
import sys
from scipy import stats
from CV import KFold
from CV import shuffle

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
sigma_sq = 1e1
reg_type = "l2"
reg_var = "1/\\sigma^2_w"

F1_scores_validation = []
F1_scores_train = []

#[kira]: shuffle data
data, target = data_processing("Bank_Data_Train.csv")
data = add_categorical(train=data,feature_str='FICO Range')
data = add_categorical(train=data,feature_str='Loan Purpose')

#k-fold
k = 5

#array_idx = 0
#penalty
penalty_set = np.arange(10,100,10)

#record accuracy score
SVC_accuracy_validation = np.zeros((len(penalty_set),k))
SVC_accuracy_train = np.zeros((len(penalty_set),k))

#if we have data recorded
penalty_idx = 0
plt.figure()
for penalty in penalty_set:
    #always shuffle data before algorithm
    shuffled_data, shuffled_target = shuffle(data,target)
    
    #k-fold devision of data
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
            data_train = pd.concat([ shuffled_data.iloc[0:indice[kfold_idx],:] , shuffled_data.iloc[indice[kfold_idx+1]:,] ])
            target_train = np.concatenate((shuffled_target[0:indice[kfold_idx]] , shuffled_target[indice[kfold_idx+1]:]))
        
        # Describe classifier and regularization type
        model = svm.SVC(C=penalty,probability=False)
        # Train model
        model.fit(X=data_train, y=target_train)
        # Predicted probabilities of label +1
        #     0.5 is an arbitrary number
        y_pred_validation = model.predict(data_validation)
        y_pred_train = model.predict(data_train)
        
        f1score_validation = f1_score(target_validation, y_pred_validation)        
        f1score_train = f1_score(target_train, y_pred_train)
        
        #get accuracy of training and validation set
        SVC_accuracy_train[penalty_idx][k_fold_idx] = f1score_train
        SVC_accuracy_validation[penalty_idx][k_fold_idx] = f1score_validation
        
        print(k_fold_idx)
        k_fold_idx += 1

        
    print(penalty)
    #plot

    penalty_idx += 1
    
plt.plot(penalty_set,np.average(SVC_accuracy_train,axis=1),color='r')
plt.plot(penalty_set,np.average(SVC_accuracy_validation,axis=1),color='b')
plt.axes().set_xscale('log')
plt.title('Learning Curve')
plt.xlabel('Threshold')
plt.ylabel('Average F1 score')
#plt.legend(loc="lower right")
plt.savefig("Q2(v)_tune.png")
