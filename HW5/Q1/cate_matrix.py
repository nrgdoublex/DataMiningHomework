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
movie_cate = []

#extract sub dataframe 
for movie in movie_list:
    print(movie)
    for row in range(0,df_cate.shape[0]):
        name = df_cate.iloc[row]['Movie_name']
        if movie == name:
            movie_cate.append(df_cate.iloc[row])

df_movie = pd.DataFrame(movie_cate)
df_movie.to_csv("movie_in_training.csv",index=False)
