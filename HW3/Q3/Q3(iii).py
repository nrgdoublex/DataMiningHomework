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
from CV import shuffle_1

#parameters for LR
sigma_sq = 0.01
reg_type = 'l2'



#read training and validation data
df = pd.read_csv('movies_training_cosine.csv')

#threshold
classify_thres = 0.4
#size of validation set
validation_size = 5000
training_size_set = np.arange(1000,25000,1000)
#record validation result
validation_result = np.zeros(len(training_size_set))

shuffled_data = shuffle_1(df)
data_validation = shuffled_data.iloc[:validation_size,:]
data_train = shuffled_data.iloc[validation_size:,:]


target_validation = data_validation['target_movie_Pulp_Fiction_1994']
data_validation = data_validation.iloc[:,1:(data_validation.shape[1]-1)]

#learn with different training sizes
training_size_idx = 0
for training_size in training_size_set:
    #shuffle original training set each time
    shuffled_training_set = shuffle_1(data_train)
    training_data = shuffled_training_set.iloc[:training_size,:]
    
    target = training_data['target_movie_Pulp_Fiction_1994']
    data = training_data.iloc[:,1:(data_train.shape[1]-1)]
    
    #build Model
    model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
    
    #fit model    
    model.fit(X=data, y=target)
    #predict on validation set
    y_pred_validation = (model.predict_proba(data_validation)[:,1] > classify_thres) + 0
    
    #f1 score
    f1score = f1_score(target_validation, y_pred_validation)
    
    validation_result[training_size_idx] = f1score
    print(f1score)
    training_size_idx += 1

plt.plot(training_size_set, validation_result)
plt.title('Learning Curve of different training size')
plt.xlabel('Size of training set')
plt.ylabel('F1 score')
plt.savefig("Q3(iii).png")