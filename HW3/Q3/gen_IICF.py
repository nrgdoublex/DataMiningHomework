import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

similarity = 'pearson'

# read data
data = pd.read_csv("movies_training.csv")

# get rid of user id and target label temporarily
data_trim = data.drop('user_id',axis=1)
data_trim = data_trim.drop('target_movie_Pulp_Fiction_1994',axis=1)

# output dataframe
data_ism = pd.DataFrame(index=data_trim.columns,columns=data_trim.columns)

# dimension
dim = data_ism.shape[1]

# get similarity
if similarity == 'cosine':
    data_ism = pd.DataFrame(index=data_trim.columns,columns=data_trim.columns)
    for i in range(0,data_ism.shape[1]) :
        print("%d" %i)
        # Loop through the columns for each column
        for j in range(0,data_ism.shape[1]) :
          # Fill in placeholder with cosine similarities
          data_ism.iloc[i,j] = cosine(data_trim.iloc[:,i],data_trim.iloc[:,j])
elif similarity == 'pearson':
    data_ism = data_trim.corr('pearson')
        
if similarity == 'cosine':
    data_ism.to_csv("cosine_similarity.csv",index=False)
elif similarity == 'pearson':
    data_ism.to_csv("pearson_similarity.csv",index=False)