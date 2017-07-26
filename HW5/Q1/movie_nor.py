import numpy as np
import pandas as pd

df = pd.read_csv("movies_training.csv")

df_sub = df.loc[:1000,:]
df_sub_m = df_sub.as_matrix()

df_out = pd.DataFrame(0,index=np.arange(len(df_sub)),columns=df_sub.columns)
df_out['user_id'] = df_sub['user_id']
df_out['target_movie_Pulp_Fiction_1994'] = df_sub['target_movie_Pulp_Fiction_1994'] - 0.5

for i in range(0,len(df_sub)):
    row = df_sub_m[i,1:500]
    mean = np.sum(row)/np.count_nonzero(row)
    df_out.iloc[i,1:500] = [ j - mean if j != 0 else 0 for j in row]

df_out.to_csv("movies_training_nor_test.csv",index=False)
