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
df_train = pd.read_csv("movies_training_normalized.csv")
df_train_trim = df_train.drop('user_id',axis=1)
matrix_train = df_train_trim.as_matrix()
matrix_train = likeorunlike_vec(matrix_train)

matrix_train_rated = nonzero_vec(matrix_train)
#read movie-category data
df_cate = pd.read_csv("movie_in_training.csv")
df_cate = df_cate.drop("Movie_name",axis=1)
matrix_cate = df_cate.as_matrix()

#each entry indicates how many times a user like a genre
user_cate = np.dot(matrix_train,matrix_cate)
#each entry indicates how many times a user has rated a genre
user_cate_rated = np.dot(matrix_train_rated,matrix_cate)
user_cate_rated = divide1_vec(user_cate_rated)
#ratio of movies in a genre one likes to movies in the genre 
user_cate_nor = np.divide(user_cate,user_cate_rated)

df_out = pd.DataFrame(user_cate_nor,columns = df_cate.columns.values)
df_out = pd.concat([df_train['user_id'],df_out],axis=1)

df_out.to_csv("user_category.csv",index=False)
