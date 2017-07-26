from __future__ import print_function
from __future__ import division
import pandas as pd
import pylab as pl
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import sys
from six import string_types
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn import svm
#from sklearn.utils import shuffle
from CV import train_test_split
from CV import shuffle
from scipy import stats


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
#SVM parameter
slack_SVM = 40

#[kira]: shuffle data
data, target = data_processing("Bank_Data_Train.csv")
data = add_categorical(train=data,feature_str='FICO Range')
data = add_categorical(train=data,feature_str='Loan Purpose')

algorithm_set = ['LR','SVC']

#main algorithm
plt.figure()
for algorithm in algorithm_set:
    #always shuffle data before algorithm
    shuffled_data, shuffled_target = shuffle(data,target)
    
    data_train, data_validation = train_test_split(shuffled_data, test_size=0.2)
    target_train, target_validation = train_test_split(shuffled_target, test_size=0.2)
    print(data_train)
    # Describe classifier and regularization type
    if algorithm == 'LR':
        model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
    else:
        model = svm.SVC(C=slack_SVM,probability=False)
    # Train model
    validation_score = model.fit(X=data_train, y=target_train).decision_function(data_validation)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr, tpr, _ = roc_curve(target_validation, validation_score)
    roc_auc = auc(fpr, tpr)
    
    lw = 2
    if algorithm == 'LR':
        plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve for LR(area = %0.2f)' % roc_auc)
    else:
        plt.plot(fpr, tpr, color='green',
             lw=lw, label='ROC curve for SVM(area = %0.2f)' % roc_auc)
        
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of Logistic Regression and soft margin SVM')
plt.legend(loc="lower right")
plt.savefig("Q2(vi).png")
    