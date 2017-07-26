import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


def cosine_sim(col1,col2):
    return cosine(col1,col2)

similarity = 'cosine'

#read data
data = pd.read_csv("movies_training.csv")
#get rid of user id and target label temporarily
data_trim = data.drop('user_id',axis=1)
data_trim = data_trim.drop('target_movie_Pulp_Fiction_1994',axis=1)
if similarity == 'cosine':
    data_ism = pd.DataFrame(index=data_trim.columns,columns=data_trim.columns)
    for i in range(0,len(data_ism.columns)) :
        print("%d" %i)
        # Loop through the columns for each column
        for j in range(0,len(data_ism.columns)) :
          # Fill in placeholder with cosine similarities
          data_ism.ix[i,j] = cosine(data_trim.ix[:,i],data_trim.ix[:,j])
elif similarity == 'pearson':
    data_ism = data_trim.corr('pearson')

data_test = pd.read_csv("movies_test_features_utf8.csv")
data_output = pd.DataFrame(0,index=np.arange(len(data_test)),columns=data.columns)
data_output['user_id'] = data_test['user_id']

data_test = data_test.drop('user_id',axis=1)
data_test = data_test.drop('target_movie_Pulp_Fiction_1994',axis=1)
#initialize output

#fill in missing data
col_num = len(data_test.columns)
row_num = len(data_test)
for row in range(0,row_num):
    print("%d" %row)
    for col in range(0,col_num):
        #missing data
        if data_test.ix[row,col] == 0:
            data_output.ix[row,col+1] = np.inner(data_ism.ix[col,:],data_test.ix[row,:]) / np.sum(np.abs(data_ism.ix[col,:])) 
        #not missing data
        else:
            data_output.ix[row,col+1] = data_test.ix[row,col]
            

if similarity == 'cosine':
    data_output.to_csv("movies_test_features_cosine.csv",index=False)
elif similarity == 'pearson':
    data_output.to_csv("movies_test_features_pearson.csv",index=False)
