Description: Given movie rating dataset, use sparse matrix SVD and PARAFAC to predict who likes Pulp Fiction movie.

Contents:
Q2(i).py: Use a sparse matrix SVD implementation to find the SVD decomposition of the user × movie rating matrix and find top 10 sigular values.
Q2(iv)_normalize.py: Normalize the dataset for future SVD procedure.
Q2(iv)_SVD_validation.py: Use cross validation to find suitable number of singular values in SVD.
Q2(iv)_SVD.py: Perform SVD and use it to predict who like Pulp Fiction movie.
Q2(v)_tensor.py: Convert raw dataset into 3-mode tensor for PARAFAC.
Q2(v)_validation.py: Use cross validation to decide the suitable rank for PARAFAC
Q2(v)_predict.py: Perform PARAFAC with optimal rank and then predict who like Pulp Fiction movie.
Q2(vi).py: Predict the actual rating each user would give to Pulp Fiction movie by PARAFAC.
