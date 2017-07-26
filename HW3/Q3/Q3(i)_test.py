from __future__ import print_function
from __future__ import division
import pandas as pd
import pylab as pl
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
#from sklearn.utils import shuffle
from CV import KFold
from CV import train_test_split
import csv
import random
import matplotlib.pyplot as plt
import sys
from CV import shuffle

#parameters for LR
sigma_sq = 0.01
reg_type = 'l2'

#read test data
df_test = pd.read_csv('movies_test_features_cosine.csv')
#read training and validation data
df = pd.read_csv('movies_training_cosine.csv')

#threshold
classify_thres = 0.5

target_train = df['target_movie_Pulp_Fiction_1994']
data_train = df.iloc[:,1:(df.shape[1]-1)]
target_test = df_test['target_movie_Pulp_Fiction_1994']
data_test = df_test.iloc[:,1:(df_test.shape[1]-1)]

#build Model
model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)

#fit model    
model.fit(X=data_train, y=target_train)

y_pred_validation = (model.predict_proba(data_test)[:,1] > classify_thres) + 0

output = pd.DataFrame(0,index=np.arange(len(data_test)),columns=['user_id','target_movie_Pulp_Fiction_1994'])
output['user_id'] = df_test['user_id']
output['target_movie_Pulp_Fiction_1994'] = y_pred_validation
output.to_csv("Q3(i)_cosine.csv",index=False)