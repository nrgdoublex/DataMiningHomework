# coding=utf_8
from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.sparse.linalg as linalg
import sys
import pandas as pd

#read train data
df_train = pd.read_csv("movies_training.csv")
df_train_trim = df_train.iloc[:,1:(df_train.shape[1]-1)]
train_len = len(df_train)

#read test data
df_test = pd.read_csv("movies_test_features.csv")
df_test_trim = df_test.iloc[:,1:(df_test.shape[1]-1)]
test_len = len(df_test)

#combine data
df_combined = df_train_trim.append(df_test_trim)
df_combined = df_combined[df_train_trim.columns]
for i in df_combined:
    col = df_combined[i]
    sum = np.sum(col)
    nonzero_count = np.count_nonzero(col)
    mean = sum/nonzero_count
    nonzero = df_combined[i].apply(lambda x: 1 if x != 0 else 0)
    mean = np.multiply(mean,nonzero)
    df_combined[i] = df_combined[i] - mean

#record mean of pulp in test data
pulp_nonzero = np.count_nonzero(df_train['target_movie_Pulp_Fiction_1994'])
pulp_sum = np.sum(df_train['target_movie_Pulp_Fiction_1994'])
pulp_mean = pulp_sum/pulp_nonzero

#output train data
df_output_train = pd.DataFrame(0,index=np.arange(len(df_train)),columns=df_train.columns)
df_output_train['user_id'] = df_train['user_id']
df_output_train['target_movie_Pulp_Fiction_1994'] = df_train['target_movie_Pulp_Fiction_1994'] - pulp_mean
df_output_train.iloc[:,1:(df_train.shape[1]-1)] = df_combined.iloc[:train_len,:]
df_output_train.to_csv("movies_training_nor.csv",index=False)

#output train data
df_output_test = pd.DataFrame(0,index=np.arange(len(df_test)),columns=df_test.columns)
df_output_test['user_id'] = df_test['user_id']
df_output_test['target_movie_Pulp_Fiction_1994'] = 0
df_output_test.iloc[:,1:(df_test.shape[1]-1)] = df_combined.iloc[train_len:,:]
df_output_test.to_csv("movies_test_features_nor.csv",index=False,encoding='utf-8')
