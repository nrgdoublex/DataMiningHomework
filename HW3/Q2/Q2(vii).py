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
#from sklearn.utils import resample
#from sklearn.utils import shuffle
from sklearn import svm
from scipy import stats
from CV import train_test_split
from CV import resample
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

def add_categorical(train, validation, feature_str):
    # encode categorical features
    encoded = pd.get_dummies(pd.concat([train[feature_str],validation[feature_str]], axis=0))#, dummy_na=True)
    train_rows = train.shape[0]
    train_encoded = encoded.iloc[:train_rows, :]
    validation_encoded = encoded.iloc[train_rows:, :] 

    train_encoded_wnoise = train_encoded.applymap(add_noise)
    validation_encoded_wnoise = validation_encoded.applymap(add_noise)

    train.drop(feature_str,axis=1, inplace=True)
    validation.drop(feature_str,axis=1, inplace=True)

    train = train.join(train_encoded_wnoise.ix[:, :])
    validation = validation.join(validation_encoded_wnoise.ix[:, :])

    return (train,validation)


# divide data into training and validation set
train_file = "Bank_Data_SeparateTrain.csv"
validation_file = "Bank_Data_SeparateValidation.csv"

# L2 Regularization penalty \\sigma^2/\\sigma^2_\\beta = sigma_sq
sigma_sq = 1e-2
reg_type = "l2"
reg_var = "1/\\sigma^2_w"

#[kira]: data processing
with open ("Bank_Data_Train.csv") as f:
    data=list(csv.reader(f))
copy = data[1:]
random.shuffle(copy)
data[1:] = copy

#test size
test_size = 0.2
#[kira]:save data
split_idx = int(np.floor(len(copy) * test_size))
with open(validation_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(data[0])
    writer.writerows(data[1:split_idx+1])
with open(train_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(data[0])
    writer.writerows(data[split_idx+1:])
    
data_train, target_train = data_processing(train_file)
data_validation, target_validation = data_processing(validation_file)
data_train, data_validation = add_categorical(train=data_train,validation=data_validation,feature_str='FICO Range')
data_train, data_validation = add_categorical(train=data_train,validation=data_validation,feature_str='Loan Purpose')

#save AUC
num_repetition = 1000
roc_auc = np.zeros(num_repetition)
for repeated_count in range(0,num_repetition):
    bs_data_train, bs_target_train = resample(data_train, target_train)
    model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
    # Train model
    validation_score = model.fit(X=bs_data_train, y=bs_target_train).decision_function(data_validation)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    
    fpr, tpr, _ = roc_curve(target_validation, validation_score)
    roc_auc[repeated_count] = auc(fpr, tpr)


roc_auc = sorted(roc_auc)
fit = stats.norm.pdf(roc_auc, np.mean(roc_auc), np.std(roc_auc))  #this is a fitting indeed
plt.plot(roc_auc,fit,'-o')
plt.hist(roc_auc,normed=False,bins=20)
plt.ylabel("Number of times")
plt.xlabel("AUC score")
plt.savefig("Q2(vii).png")