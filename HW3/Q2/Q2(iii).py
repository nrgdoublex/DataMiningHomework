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
training_size_array = [50,100,300,600,1000,1500,2000,2500]  # various training size
threshold = 0.5                                             # for logistic regression
repeat_times = 10                                          # repeat the same experiments

# L2 Regularization penalty \\sigma^2/\\sigma^2_\\beta = sigma_sq
sigma_sq = 1e-2
reg_type = "l2"
reg_var = "1/\\sigma^2_w"  

# store f1 scores
F1_scores_validation = np.zeros((len(training_size_array),repeat_times))
F1_scores_train = np.zeros((len(training_size_array),repeat_times))

# read csv file
df = pd.read_csv(data_file)

for idx ,training_size in enumerate(training_size_array):
    for repeat_idx in range(repeat_times):
        print repeat_idx
        # shuffle original DataFrame for future split
        df.index = np.random.permutation(df.index)
        df.sort_index(axis=0,inplace=True)
        
        # split dataframe into training and test set
        df_train = df.iloc[:training_size,:].copy()
        df_valid = df.iloc[training_size:,:].copy()
        
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
        F1_scores_train[idx][repeat_idx] = f1score_train
        F1_scores_validation[idx][repeat_idx] = f1score_validation
        
# plot
plt.plot(training_size_array,np.average(F1_scores_train,axis=1),label='training')
plt.plot(training_size_array,np.average(F1_scores_validation,axis=1),label='test')
plt.title('Learning Curve')
plt.xlabel('number of training samples')
plt.ylabel('Mean F1 score')
plt.legend(loc="upper right")
plt.savefig("Q2(iii).png")