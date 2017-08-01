import numpy as np
import pandas as pd

#read training data
df = pd.read_csv("movies_training.csv")

#list to store movie name
movie_list = []
 
#extract movie name
for movie in df.columns:
    if movie != 'user_id':
        movie = movie.replace("movie_rating_",'')
        movie = movie.replace("target_movie_",'')
        movie_list.append(movie)


df_cate = pd.read_csv("movie_categories.csv")

#extract sub dataframe 
index = map(lambda x: x in movie_list, df_cate['Movie_name'])
df_movie = df_cate.iloc[index,:]
df_movie.set_index('Movie_name',inplace=True)
df_movie = df_movie.reindex(movie_list)
df_movie.to_csv("movie_in_training.csv")
