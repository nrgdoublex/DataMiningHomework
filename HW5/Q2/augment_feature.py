import pandas as pd
import numpy as np

num_augment = 11

in_file = "movies_training.csv"
out_file = "movies_training_aug.csv"
data = pd.read_csv(in_file)

features = data.columns.values
df_out = data[np.delete(features,-1)]

count = 0
for i in range(1,num_augment+1):
    for j in range(i,num_augment+1):
        feature1 = features[i]
        feature2 = features[j]
        com_feature = feature1 + ' * ' + feature2
        print(com_feature)
        df_out[com_feature] = data[feature1] * data[feature2]
        count += 1

df_out['target_movie_Pulp_Fiction_1994'] = data['target_movie_Pulp_Fiction_1994']
df_out.to_csv(out_file,index=False)