# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import scipy.stats as stats
from clean_data import data_processing
from clean_data import add_categorical

def one_tailed_ttest(data1,data2):
    len_data1 = len(data1)
    len_data2 = len(data2)
    mean_data1 = np.average(data1)
    mean_data2 = np.average(data2)
    var_data1 = np.var(data1)
    var_data2 = np.var(data2)
     
    t_score = (mean_data1-mean_data2) / np.sqrt(var_data1/len_data1+var_data2/len_data2)
    df = (np.square(var_data1/len_data1+var_data2/len_data2) / 
        (np.square(var_data1/len_data1)/(len_data1-1) + np.square(var_data2/len_data2)/(len_data2-1)))
    p_value = stats.t.sf(t_score, df)
     
    return (t_score,p_value)

# parameters
data_file = 'Bank_Data_Train.csv'                           # original data
LR_threshold = 0.35                                         # for logistic regression
k_fold = 20                                                 # k-fold parameter
LR_f1_validation = np.zeros(k_fold)
LR_f1_train = np.zeros(k_fold)
SVC_f1_validation = np.zeros(k_fold)
SVC_f1_train = np.zeros(k_fold)
algorithm_set = ['LR','SVC']
penalty_svm = 40                                            #penalty of svm
confidence_level = 0.05

# L2 Regularization penalty \\sigma^2/\\sigma^2_\\beta = sigma_sq
sigma_sq = 1e-2
reg_type = "l2"

# read csv file
df = pd.read_csv(data_file)

# use k-fold CV to decide suitable threshold
for idx, algorithm in enumerate(algorithm_set):
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
        if algorithm == 'LR':
            model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
        else:
            model = SVC(C=penalty_svm,probability=False)
        # Train model
        model.fit(X=data_train, y=target_train)
        
        # Predicted probabilities of label +1
        #     0.5 is an arbitrary number
        
        # linear regression
        if algorithm == "LR":
            y_pred_validation = (model.predict_proba(data_validation)[:,1] > LR_threshold) + 0
            y_pred_train = (model.predict_proba(data_train)[:,1] > LR_threshold) + 0
        else:
            y_pred_validation = model.predict(data_validation)
            y_pred_train = model.predict(data_train)
        
        # f1 score
        f1score_validation = f1_score(target_validation, y_pred_validation)
        f1score_train = f1_score(target_train, y_pred_train)
        
        #get accuracy of training and validation set
        if algorithm == 'LR':
            LR_f1_train[part] = f1score_train
            LR_f1_validation[part] = f1score_validation
        else:
            SVC_f1_train[part] = f1score_train
            SVC_f1_validation[part] = f1score_validation
            
(t_score,p_value) = one_tailed_ttest(LR_f1_validation, SVC_f1_validation)
if p_value < confidence_level:
    if t_score > 0:
        print("LR is more accurate")
    else:
        print("SVM is more accurate")
else:
    print("we cannot differentiate the power of LR and SVC")