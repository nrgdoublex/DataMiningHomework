import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

# file names
training_file = 'movies_training_IICF.csv'

# parameters for LR
sigma_sq = 0.01
reg_type = 'l2'

# read training data
df = pd.read_csv(training_file)
total_size = df.shape[0]

# threshold
classify_thres = 0.4

# size of validation set
validation_size = 5000
training_size_set = np.arange(1000,total_size-validation_size,1000)

# record validation result
validation_result = np.zeros(len(training_size_set))

# shuffle original DataFrame for future split
df.index = np.random.permutation(df.index)
df.sort_index(axis=0,inplace=True)

# split into training and validation
df_validation = df.iloc[:validation_size,:].copy()
df_train = df.iloc[validation_size:,:].copy()

# reindex to make future shuffling easy
df_train.index = np.arange(0,df_train.shape[0])

# split to data and target
target_validation = df_validation['target_movie_Pulp_Fiction_1994']
data_validation = df_validation.iloc[:,1:(df_validation.shape[1]-1)]

#learn with different training sizes
for idx, training_size in enumerate(training_size_set):
    #shuffle original training set each time
    df_train.index = np.random.permutation(df_train.index)
    df_train.sort_index(axis=0,inplace=True)
    
    # sample training data
    training_data = df_train.iloc[:training_size,:]
    
    target = training_data['target_movie_Pulp_Fiction_1994']
    data = training_data.iloc[:,1:(df_train.shape[1]-1)]
    
    # build Model
    model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
    
    # fit model    
    model.fit(X=data, y=target)
    #predict on validation set
    y_pred_validation = (model.predict_proba(data_validation)[:,1] > classify_thres) + 0
    
    # f1 score
    f1score = f1_score(target_validation, y_pred_validation)
    
    validation_result[idx] = f1score
    print "Training size = {0}, F1 score = {1}".format(training_size, f1score)

# plot
plt.plot(training_size_set, validation_result)
plt.title('Learning Curve of different training size')
plt.xlabel('Size of training set')
plt.ylabel('F1 score')
plt.savefig("Q3(iii).png")