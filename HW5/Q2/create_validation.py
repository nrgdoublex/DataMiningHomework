import pandas as pd
import matplotlib.pylab as pl
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

size = 25000

df = pd.read_csv("movies_training.csv")

df_valid = df.iloc[size:,]
df_valid.to_csv("movies_validation.csv",index=False)