from __future__ import print_function
from __future__ import division
import pandas as pd
import pylab as pl
import numpy as np
import random
import scipy.stats as stats

def KFold(df,k):
    quan = np.zeros(k+1)
    total_size = len(df)
    subset_size = np.floor(total_size / k)
    residue = np.remainder(total_size,k)
    end_idx = 0
    for i in range(1,k+1):
        if i <= residue:
            quan[i] = end_idx + subset_size + 1
            end_idx += (subset_size + 1)
        else:
            quan[i] = end_idx + subset_size
            end_idx += subset_size
    return [int(i) for i in quan]
    
def train_test_split(df,test_size):
    if test_size > 1 or test_size < 0:
        return
    total_size = len(df)
    split_idx = int(np.floor(total_size * test_size))
    if split_idx == 0:
        test = []
        train = df
    elif split_idx == len(df):
        test = df
        train = []
    else:
        test = df[:split_idx]
        train = df[split_idx:]
        
    return (train,test)

def shuffle(df, array):   
    shuffled_array = np.zeros(len(df),dtype=np.int32)
    
    randomize = np.arange(len(df))
    np.random.shuffle(randomize)
    shuffled_df = df.iloc[randomize]
    for i in range(0,len(randomize)):
        shuffled_array[i] = array[randomize[i]]
    return (shuffled_df,shuffled_array)

def shuffle_1(df):   
    randomize = np.arange(len(df))
    np.random.shuffle(randomize)
    shuffled_df = df.iloc[randomize]

    return shuffled_df

def resample(df,array):
    total_size = len(df)
    indices = np.zeros(total_size,dtype=np.int32)
    for i in range(0,total_size):
        indices[i] = random.randint(0,total_size-1)
    output = pd.DataFrame(index=np.arange(total_size),columns=df.columns)
    output2 = np.zeros(total_size,dtype=np.int32)
    
    output = df.iloc[indices]
    for i in range(0,total_size):
        output2[i] = array[indices[i]]
    return (output,output2)

def one_tailed_ttest(data1,data2):
    len_data1 = len(data1)
    len_data2 = len(data2)
    mean_data1 = np.average(data1)
    mean_data2 = np.average(data2)
    var_data1 = np.var(data1)
    var_data2 = np.var(data2)
    
    t_score = (mean_data1-mean_data2) / np.sqrt(var_data1/len_data1+var_data2/len_data2)
    df = np.square(var_data1/len_data1+var_data2/len_data2) / \
        (np.square(var_data1/len_data1)/(len_data1-1) + np.square(var_data2/len_data2)/(len_data2-1))
    p_value = stats.t.sf(t_score, df)
    
    return (t_score,p_value)