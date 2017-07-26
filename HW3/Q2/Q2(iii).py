from __future__ import print_function
from __future__ import division
import pandas as pd
import pylab as pl
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import csv
import random
import matplotlib.pyplot as plt
import sys

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


train_file = "Bank_Data_SeparateTrain.csv"
validation_file = "Bank_Data_SeparateValidation.csv"

# L2 Regularization penalty \\sigma^2/\\sigma^2_\\beta = sigma_sq
sigma_sq = 1e-2
reg_type = "l2"
reg_var = "1/\\sigma^2_w"

F1_scores_validation = []
F1_scores_train = []

#[kira]: shuffle data
with open ("Bank_Data_Train.csv") as f:
    data=list(csv.reader(f))
total_size = len(data)
repeat_times = 100
training_size = [50,100,300,600,1000,1500,2000,2500]
accuracy_validation = np.zeros((repeat_times,len(training_size)))
accuracy_train = np.zeros((repeat_times,len(training_size)))

#classification threshold
threshold = 0.5
copy = data[1:]
#[kira]:for each training size, get F1 score
for repeat_idx in range(0,repeat_times):
    array_idx = 0
    for size in training_size:
        random.shuffle(copy)
        data[1:] = copy
        
        #[kira]:divide data
        with open(train_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data[0])
            writer.writerows(data[1:size+1])
        with open(validation_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data[0])
            writer.writerows(data[size+1:])
        
        data_train, target_train = data_processing(train_file)
        data_validation, target_validation = data_processing(validation_file)

        
        # replace categorical strings with 1-of-K coding and add a small amount of Gaussian noise so it follows Gaussian model assumption
        
        data_train, data_validation = add_categorical(train=data_train,validation=data_validation,feature_str='FICO Range')
        
        data_train, data_validation = add_categorical(train=data_train,validation=data_validation,feature_str='Loan Purpose')
        # Describe classifier and regularization type
        logr = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
        # Train model
        logr.fit(X=data_train, y=target_train)
        # Predicted probabilities of label +1
        #     0.5 is an arbitrary number
        y_pred_validation = (logr.predict_proba(data_validation)[:,1] > threshold) + 0
        y_pred_train = (logr.predict_proba(data_train)[:,1] > threshold) + 0
        
        f1score_validation = f1_score(target_validation, y_pred_validation)
        #F1_scores_validation.append(f1score_validation)
            
        f1score_train = f1_score(target_train, y_pred_train)
        #F1_scores_train.append(f1score_train)
        
        #get accuracy of training and validation set
        accuracy_train[repeat_idx][array_idx] = f1score_train
        accuracy_validation[repeat_idx][array_idx] = f1score_validation
        
        array_idx += 1

# plot

plt.plot(training_size,np.average(accuracy_train,axis=0),label='training')
plt.plot(training_size,np.average(accuracy_validation,axis=0),label='testing')
plt.title('Learning Curve')
plt.xlabel('number of training samples')
plt.ylabel('Mean F1 score')
plt.legend(loc="upper right")
plt.savefig("Q2(iii).png")