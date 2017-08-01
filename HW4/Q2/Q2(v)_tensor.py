from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.sparse.linalg as linalg
import sys
import pandas as pd
import sklearn.metrics

tensor = []

#read user-movie data
df = pd.read_csv("movies_training_nor.csv")
df_test = pd.read_csv("movies_test_features_nor.csv")


#read gener data
df_genre = pd.read_csv("movie_categories.csv")
row = df_genre[ df_genre['Movie_name'] =='Pulp_Fiction_1994' ]

#build dictionary of movie:col_num
dict_movie = {}
dict_genre = {}
movie_set = df.columns
for i in range(1,df.shape[1]):
    movie_name = movie_set[i].replace('movie_rating_',"").replace('target_movie_',"")
    dict_movie[i] = movie_name
    dict_genre[movie_name] = df_genre[df_genre['Movie_name'] == movie_name].as_matrix().reshape(-1)[1:]


#generate tensor
row_num = df.shape[0]
col_num = df.shape[1]
user_idx = 1
#i is the index of user
for i in range(0,row_num):
    row = df.iloc[i,:]
    #j is the index of movie
    for j in range(1,col_num):
        rating = row[j]
        if rating == 0:
            continue
        movie_name = dict_movie[j]
        for k in range(0,len(dict_genre[movie_name])):
            #find genre
            if dict_genre[movie_name][k] == 1:
                #print([i+1,j,k+1,row[j]])
                tensor.append([i+1,j,k+1,rating])
#output tensor
with open('tensor_train.csv','w') as f:
    for idx in range(0,len(tensor)):
        i,j,k,value = tensor[idx]
        f.write("%d,%d,%d,%f\n" %(i,j,k,value))
         
tensor_test = []
row_num = df_test.shape[0]
col_num = df_test.shape[1]
user_idx = 1
#i is the index of user
for i in range(0,row_num):
    row = df_test.iloc[i,:]
    #j is the index of movie
    for j in range(1,col_num):
        rating = row[j]
        if rating == 0:
            continue
        movie_name = dict_movie[j]
        for k in range(0,len(dict_genre[movie_name])):
            #find genre
            if dict_genre[movie_name][k] == 1:
                #print([i+1,j,k+1,row[j]])
                tensor_test.append([i+30000,j,k+1,rating])
         
#output tensor
with open('tensor_test.csv','w') as f:
    for idx in range(0,len(tensor_test)):
        i,j,k,value = tensor_test[idx]
        f.write("%d,%d,%d,%f\n" %(i,j,k,value)) 
#np.savetxt('tensor.csv',tensor,fmt='%.6f',delimiter=',',newline='\n')
