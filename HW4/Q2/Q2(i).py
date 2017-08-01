from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.sparse.linalg import *
import sys
import pandas as pd

df = pd.read_csv("movies_training.csv")
df_trim = df.iloc[:,1:(df.shape[1]-1)]

# SVD
U, sigma, V = svds(df_trim,k=20)

sigma_sorted = sorted(sigma,reverse=True)
print(sigma_sorted)
