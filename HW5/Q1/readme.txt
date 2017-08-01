Description: Use K-means algorithm to cluster users.

Contents:
movie_normalize.py: normalize features so that missing values now represent the mean behavior.
extract_movie_category.py: from movie_categories.csv extract movies we are interested.
gen_labels: decide which user likes which kind of movies most, and generate corresponding label for that user.
k_means_CV.py: Use cross validation to decide the best number of clusters for our data.
k_means.py: Use k-means algorithm to cluster data.
CV.py: helper functions in cross validation and shuffling data
