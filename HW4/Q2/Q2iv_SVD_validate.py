from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.sparse.linalg as linalg
import sys
import pandas as pd
import sklearn.metrics
from CV import KFold
from CV import shuffle_1

#read train data
df_train = pd.read_csv("movies_training_nor.csv")
df_train_trim = df_train.iloc[:,1:(df_train.shape[1])]
df_train_trim_1 = df_train_trim.copy()

 
#read test data
df_test = pd.read_csv("movies_test_features_nor.csv")
df_test_trim = df_test.iloc[:,1:(df_test.shape[1])]
test_len = len(df_test)
 
#k-fold cross validation
fold = 10 
 
#number of singular values
num_singular = np.arange(1,11)

#cross validation
for num in num_singular:
    print("number of singular value = %d" %num)
    #keep it, don't tailor it
    shuffled_df_train = shuffle_1(df_train_trim)
    indice = KFold(shuffled_df_train, fold)
    score_array = np.zeros(fold)
    for kfold_idx in range(0,fold):
        shuffled_df_train_copy = shuffled_df_train.copy()
        shuffled_df_train_copy.iloc[indice[kfold_idx]:indice[kfold_idx+1],shuffled_df_train_copy.shape[1]-1] = 0
        #SVD
        U, sigma, V = linalg.svds(shuffled_df_train_copy,k=num,which='LM')
        m = np.zeros((U.shape[0], V.shape[1]))
        for i in range(0,len(sigma)):
            m += sigma[i] * np.outer(U.T[i], V[i])
        predict = m[indice[kfold_idx]:indice[kfold_idx+1],m.shape[1]-1]
        predict = [1 if i > 0 else 0 for i in predict]
        target_validation = [1 if i > 0 else 0 for i in \
                    shuffled_df_train.iloc[indice[kfold_idx]:indice[kfold_idx+1],shuffled_df_train.shape[1]-1]]
        F1_score = sklearn.metrics.f1_score(target_validation,predict)
        score_array[kfold_idx] = F1_score
    print(np.mean(score_array))
#print(sklearn.metrics.f1_score(predict[:len(df_train)],target_train[:len(df_train)]))
#print(np.sum(-np.logical_xor(predict[:len(df_train)],target_train[:len(df_train)])))