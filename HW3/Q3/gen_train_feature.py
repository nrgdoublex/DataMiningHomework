import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

similarity = "pearson"

def cosine_sim(col1,col2):
    return cosine(col1,col2)

# file names
if similarity == "cosine":
    coef_data = "cosine_similarity.csv"
else:
    coef_data = "pearson_similarity.csv"

# dataframe for cosine coefficient
data_coef = pd.read_csv(coef_data)

# dataframe for test data
data_train = pd.read_csv("movies_training.csv")
data_output = pd.DataFrame(0,index=data_train.index,columns=data_train.columns)
data_output['user_id'] = data_train['user_id']
data_output['target_movie_Pulp_Fiction_1994'] = data_train['target_movie_Pulp_Fiction_1994']
data_train = data_train.drop('user_id',axis=1)
data_train = data_train.drop('target_movie_Pulp_Fiction_1994',axis=1)

#fill in missing data
col_num = data_train.shape[1]
row_num = data_train.shape[0]
for row in range(0,row_num):
    print("%d" %row)
    for col in range(0,col_num):
        #missing data
        if data_train.iloc[row,col] == 0:
            data_output.iloc[row,col+1] = np.inner(data_coef.iloc[col,:],data_train.iloc[row,:]) / np.sum(np.abs(data_coef.iloc[col,:])) 
        #not missing data
        else:
            data_output.iloc[row,col+1] = data_train.iloc[row,col]
            

        
# output
data_output.to_csv("movies_training_IICF.csv",index=False)