from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.sparse.linalg as linalg
import sys
import pandas as pd
import sklearn.metrics

#parameter
num_sin = 6

#read train data
df_train = pd.read_csv("movies_training_nor.csv")
df_train_trim = df_train.iloc[:,1:(df_train.shape[1])]
train_len = len(df_train)
 
#read test data
df_test = pd.read_csv("movies_test_features_nor.csv")
df_test_trim = df_test.iloc[:,1:(df_test.shape[1])]
test_len = len(df_test)
 
df_combined = df_train_trim.append(df_test_trim)
df_combined = df_combined[df_train_trim.columns]
 
#SVD
U, sigma, V = linalg.svds(df_combined,k=num_sin,which='LM')
m = np.zeros((U.shape[0], V.shape[1]))
for i in range(0,len(sigma)):
    m += sigma[i] * np.outer(U.T[i], V[i])
predict = m[:,m.shape[1]-1]
predict = [1 if i > 0 else 0 for i in predict]

target_train = [1 if i > 0 else 0 for i in df_train['target_movie_Pulp_Fiction_1994']]

df_output = pd.DataFrame(index=np.arange(len(df_test)),columns=['user_id','target_movie_Pulp_Fiction_1994'])
df_output['user_id'] = df_test['user_id']
df_output['target_movie_Pulp_Fiction_1994'] = predict[train_len:]
df_output.to_csv("Q2iv_predict.csv",index=False)

#print(sklearn.metrics.f1_score(predict[:len(df_train)],target_train[:len(df_train)]))
print(np.sum(-np.logical_xor(predict[:len(df_train)],target_train[:len(df_train)])))