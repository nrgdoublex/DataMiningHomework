import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# file names
training_file = "movies_training_IICF.csv"

# parameters for LR
sigma_sq = 0.001
reg_type = 'l2'

# other parameters
classify_thres = 0.4                        # threshold for LR
k_fold = 10                                 # k-fold cross validation

#storing f1 score
LR_accuracy_validation = np.zeros(k_fold)
LR_accuracy_train = np.zeros(k_fold)

#read training and validation data
df = pd.read_csv(training_file)

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
    
    target_train = df_train['target_movie_Pulp_Fiction_1994']
    data_train = df_train.iloc[:,1:(df_train.shape[1]-1)]               # ignore 'user_id' column
    target_validation = df_valid['target_movie_Pulp_Fiction_1994']
    data_validation = df_valid.iloc[:,1:(df_valid.shape[1]-1)]          # ignore 'user_id' column

    #build Model
    model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)

    #fit model    
    model.fit(X=data_train, y=target_train)
    
    y_pred_validation = (model.predict_proba(data_validation)[:,1] > classify_thres) + 0
    
    #accuracy
    accuracy = 0
    copy_target = np.array(target_validation)
    for i in range(0,len(y_pred_validation)):
        if y_pred_validation[i] == copy_target[i]:
            accuracy += 1
    accuracy = accuracy /float(len(y_pred_validation))
    print "The accuracy of {0}-th CV = {1}".format(part+1,accuracy)
                
    #record accuracy
    LR_accuracy_validation[part] = accuracy

    
print "The mean accuracy = {0}".format(np.mean(LR_accuracy_validation))