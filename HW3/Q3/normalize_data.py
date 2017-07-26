import pandas as pd
import numpy as np

#read data
#data = pd.read_csv("test.csv")
data = pd.read_csv("movies_training.csv")
#get rid of user id and target label temporarily
data_trim = data.drop('user_id',axis=1)
data_trim = data_trim.drop('target_movie_Pulp_Fiction_1994',axis=1)

#initialize output
data_output = pd.DataFrame(0,index=np.arange(len(data)),columns=data.columns)

#normalize data
for i in range(0,len(data)):
    nonzero = np.count_nonzero(data_trim.iloc[i,:])
    if nonzero != 0:
        mean = np.sum(data_trim.iloc[i,:]) / nonzero
    else:
        mean = 0
    print(i)
    print(mean)
    for j in range(1,len(data.iloc[i,:])-1):
        if data.iloc[i,j] != 0:
            data_output.iloc[i,j] = data.iloc[i,j] - mean
        else:
            data_output.iloc[i,j] = 0


            
data_output['user_id'] = data['user_id']
data_output['target_movie_Pulp_Fiction_1994'] = data['target_movie_Pulp_Fiction_1994']

data_output.to_csv("movies_training_normalized.csv",index=False)
#data_output.to_csv("test_output.csv",index=False)