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
df_train_trim = df_train.iloc[:,1:(df_train.shape[1])].copy()

 
#read test data
df_test = pd.read_csv("movies_test_features_nor.csv")
df_test_trim = df_test.iloc[:,1:(df_test.shape[1])]
test_len = len(df_test)
 
#k-fold cross validation
k_fold = 10 
 
#number of singular values
num_singular = np.arange(1,11)

#cross validation
for num in num_singular:
    print("number of singular value = %d" %num)

    # shuffle the data
    df_train_trim.index = np.random.permutation(df_train_trim.index)
    df_train_trim.sort_index(axis=0,inplace=True)

    indice = KFold(df_train_trim, k_fold)
    score_array = np.zeros(k_fold)
    for kfold_idx in range(0,k_fold):
        df_train_copy = df_train_trim.copy()
        df_train_copy.iloc[indice[kfold_idx]:indice[kfold_idx+1],df_train_copy.shape[1]-1] = 0
        #SVD
        U, sigma, V = linalg.svds(df_train_copy,k=num,which='LM')
        m = np.zeros((U.shape[0], V.shape[1]))
        for i in range(0,len(sigma)):
            m += sigma[i] * np.outer(U.T[i], V[i])
        predict = m[indice[kfold_idx]:indice[kfold_idx+1],m.shape[1]-1]
        predict = [1 if i > 0 else 0 for i in predict]
        target_validation = [1 if i > 0 else 0 for i in \
                    df_train_trim.iloc[indice[kfold_idx]:indice[kfold_idx+1],df_train_trim.shape[1]-1]]
        F1_score = sklearn.metrics.f1_score(target_validation,predict)
        score_array[kfold_idx] = F1_score
    print(np.mean(score_array))
