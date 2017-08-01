import numpy as np
import pandas as pd

def nonzero(a):
    if a != 0:
        return 1
    else:
        return 0

def likeorunlike(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0

def divide1(a):
    if a == 0:
        return 1
    else:
        return a

nonzero_vec = np.vectorize(nonzero)
divide1_vec = np.vectorize(divide1)
likeorunlike_vec = np.vectorize(likeorunlike)

#read training data
df_train = pd.read_csv("movies_training.csv")
df_train_trim = df_train.drop('user_id',axis=1)

#read movie-category data
df_cate = pd.read_csv("movie_in_training.csv")
df_cate = df_cate.drop("Movie_name",axis=1)

#each entry indicates how many times a user like a genre
user_cate = np.dot(df_train_trim,df_cate)
#get maximum entry index
argmax = np.argmax(user_cate, axis=1)

test = np.zeros(20,dtype=np.int32)
for i in range(0,user_cate.shape[0]):
    test[argmax[i]] += 1
print(test)
print(np.sum(test))

df_out = pd.DataFrame(0,index=np.arange(0,len(df_train)),columns=['user_id','labels'],)
df_out['user_id'] = df_train['user_id']
df_out['labels'] = argmax
df_out.to_csv("labels.csv",index=False)