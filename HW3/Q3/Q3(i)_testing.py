import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

# file names
training_file = "movies_training_IICF.csv"
test_file = "movies_test_features_IICF.csv"

#parameters for LR
sigma_sq = 0.01
reg_type = 'l2'

# read training data
df_train = pd.read_csv(training_file)
# read test data
df_test = pd.read_csv(test_file)


# threshold
classify_thres = 0.4

target_train = df_train['target_movie_Pulp_Fiction_1994']
data_train = df_train.iloc[:,1:(df_train.shape[1]-1)]
target_test = df_test['target_movie_Pulp_Fiction_1994']
data_test = df_test.iloc[:,1:(df_test.shape[1]-1)]

# build Model
model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)

# fit model    
model.fit(X=data_train, y=target_train)

y_pred_validation = (model.predict_proba(data_test)[:,1] > classify_thres) + 0

output = pd.DataFrame(0,index=np.arange(len(data_test)),columns=['user_id','target_movie_Pulp_Fiction_1994'])
output['user_id'] = df_test['user_id']
output['target_movie_Pulp_Fiction_1994'] = y_pred_validation
output.to_csv("Q3(i)_submission.csv",index=False)