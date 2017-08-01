# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC 
import scipy.stats as stats
import clean_data

# parameters
data_file = 'Bank_Data_Train.csv'                           # original data
algorithm_set = ['LR','SVC']
penalty_svm = 40                                            #penalty of svm
test_size = 0.2

# L2 Regularization penalty \\sigma^2/\\sigma^2_\\beta = sigma_sq
sigma_sq = 1e-2
reg_type = "l2" 

# read csv file
df = pd.read_csv(data_file)


for idx, algorithm in enumerate(algorithm_set):
    # shuffle original DataFrame for future split
    df.index = np.random.permutation(df.index)
    df.sort_index(axis=0,inplace=True)
    
        
    # split dataframe into training and test set
    df_train = df.iloc[:int(df.shape[0]*test_size),:].copy()
    df_valid = df.iloc[int(df.shape[0]*test_size):,:].copy()
    
    # augment and clean dataframe
    data_train, target_train = clean_data.data_processing(df_train)
    data_validation, target_validation = clean_data.data_processing(df_valid)
    data_train, data_validation = clean_data.add_categorical(train=data_train,validation=data_validation,feature_str='FICO Range')
    data_train, data_validation = clean_data.add_categorical(train=data_train,validation=data_validation,feature_str='Loan Purpose')
    
    # Describe classifier and regularization type
    if algorithm == 'LR':
        model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
    else:
        model = SVC(C=penalty_svm,probability=False)
    # Train model
    model.fit(X=data_train, y=target_train)
    
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