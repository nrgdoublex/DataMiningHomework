Description: Use logistic regression to decide which user likes Pulp Fiction movie.

Contents:
augment_feature.py: Add some polynomial features to original data
create_validation.py: split part of data and save it as validation set.
LR.movie_1e-10.py: Use logistic regression with regularization parameter = 1e-10 and plot the learning curve
LR.movie_1e0.py: Use logistic regression with regularization parameter = 1 and plot the learning curve
LR.movie_cv.py: Use cross validation to choose best regularization strength.
LR.movie_cv.py: Use L1 regularization and describe how that affects the logistic regression parameters w.r.t. L2 regularization. Compare their nonzero parameters.
CV.py: helper function for cross validation.
