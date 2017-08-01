import numpy as np
import pandas as pd

# read data
df = pd.read_csv("movies_training.csv")

# get rid of unnecessary rows
df_sub = df.drop(['user_id','target_movie_Pulp_Fiction_1994'],axis=1)
# create output dataframe
df_output = pd.DataFrame(index=df_sub.index,columns=df_sub.columns)

# normalize each row
for i in df_sub.index:
    print i
    row = df_sub.iloc[i,:]
    mean = np.sum(row)/np.count_nonzero(row)
    df_output.iloc[i,:] = [ item - mean if item != 0 else 0 for item in row ]
    
# concatenate back
df_output = pd.concat([df['user_id'], df_output],axis=1)
df_output['target_movie_Pulp_Fiction_1994'] = df['target_movie_Pulp_Fiction_1994'] - 0.5

# output
df_output.to_csv("movies_training_normalized.csv",index=False)
