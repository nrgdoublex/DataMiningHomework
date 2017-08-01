# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
import clean_data

# parameters
data_file = 'Bank_Data_Train.csv'                           # original data
test_size = 0.2

# L2 Regularization penalty \\sigma^2/\\sigma^2_\\beta = sigma_sq
sigma_sq = 1e-2
reg_type = "l2" 

# read csv file
df = pd.read_csv(data_file)

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

# bootstrapping
repetition = 1000
roc_auc = np.zeros(repetition)
for idx in range(repetition):
    resample_indices = np.random.randint(0,data_train.shape[0],data_train.shape[0])
    
    bs_data_train = data_train.iloc[resample_indices,:]
    bs_target_train = [target_train[i] for i in resample_indices]
    model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
    # Train model
    validation_score = model.fit(X=bs_data_train, y=bs_target_train).decision_function(data_validation)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    
    fpr, tpr, _ = roc_curve(target_validation, validation_score)
    roc_auc[idx] = auc(fpr, tpr)

roc_auc = sorted(roc_auc)
fit = stats.norm.pdf(roc_auc, np.mean(roc_auc), np.std(roc_auc))  #this is a fitting indeed
plt.plot(roc_auc,fit,'-o')
plt.hist(roc_auc,normed=False,bins=20)
plt.ylabel("Number of times")
plt.xlabel("AUC score")
plt.savefig("Q2(vii).png")