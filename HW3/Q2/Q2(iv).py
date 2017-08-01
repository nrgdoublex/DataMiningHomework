# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from clean_data import data_processing
from clean_data import add_categorical

# parameters
data_file = 'Bank_Data_Train.csv'                           # original data
threshold_array = np.arange(0.05,0.85,0.05)                 # for logistic regression
k_fold = 10                                                 # k-fold parameter

# L2 Regularization penalty \\sigma^2/\\sigma^2_\\beta = sigma_sq
sigma_sq = 1e-2
reg_type = "l2"

# store f1 scores
F1_scores_validation = np.zeros((len(threshold_array),k_fold))
F1_scores_train = np.zeros((len(threshold_array),k_fold))

# read csv file
df = pd.read_csv(data_file)

# use k-fold CV to decide suitable threshold
for idx, threshold in enumerate(threshold_array):
    # shuffle original DataFrame for future split
    df.index = np.random.permutation(df.index)
    df.sort_index(axis=0,inplace=True)
    
    # decide which part of k-fold
    chunk_size = df.shape[0] / k_fold
    for part in range(k_fold):
        indices = np.arange(0,df.shape[0])
        
        # split indices to decide which part to training and test
        training_indices = filter(lambda x: x / chunk_size  == part, indices)
        if k_fold * chunk_size < df.shape[0] and part < df.shape[0] - k_fold * chunk_size: 
            training_indices = training_indices + [indices[-part-1]]
        validation_indices = [index for index in indices if index not in training_indices]
        
        # split dataframe into training and test set
        df_train = df.iloc[training_indices,:].copy()
        df_valid = df.iloc[validation_indices,:].copy()
        
        # augment and clean dataframe
        data_train, target_train = data_processing(df_train)
        data_validation, target_validation = data_processing(df_valid)
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
        
        # f1 score
        f1score_validation = f1_score(target_validation, y_pred_validation)
        f1score_train = f1_score(target_train, y_pred_train)
        
        #get accuracy of training and validation set
        F1_scores_train[idx][part] = f1score_train
        F1_scores_validation[idx][part] = f1score_validation
        
# plot
plt.plot(threshold_array,np.average(F1_scores_train,axis=1),label='training')
plt.plot(threshold_array,np.average(F1_scores_validation,axis=1),label='validation')
plt.title('Threshold of LR vs F1 score')
plt.xlabel('Threshold')
plt.ylabel('Average F1 score')
plt.legend(loc="lower left")
plt.savefig("Q2(iv).png")
        