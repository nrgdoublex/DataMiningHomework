# -*- coding: latin-1 -*-
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

# Use Thompson Sampling algorithm
datafile = "Q1.csv"
columnname = {'number': 'Number2013', 'percent': 'Percent2013'
           , 'number.1': 'Number2012', 'percent.1': 'Percent2012'
           , 'number.2': 'Number2004', 'percent.2': 'Percent2004'}

# read csv and clean data
df = pd.read_csv(datafile, encoding='latin-1', header=1)
df.rename(columns=columnname, inplace=True)
for column in ['Number2013','Percent2013']:
    df[column] = df[column].apply(pd.to_numeric, errors='coerce')
df['Percent2013'] = df['Percent2013'].div(100)

# Create a dataframe with the variables of interest without the invalid rows (need to replace "No Value" by "NA" in the original file)
wdf = df[['State','County','Number2013','Percent2013']].copy().dropna(axis=0)
wdf.index = xrange(wdf.shape[0])
# Number of people in county
wdf['Population2013'] =  wdf.Number2013.div(wdf.Percent2013).apply(lambda x: int(x))

# Matrix of counties x 1000 posterior samples
samples = np.zeros(shape=(wdf.shape[0],1000))
# Give index to county names
county_to_idx = defaultdict(lambda:len(county_to_idx))
# Revese lookup
idx_to_county = {}

# Beta prior (hyper)parameters, better to add strong priors towards low rates
a, b = 1, 1
for i in wdf.index:
    # Sample 1000 values from posterior distribution, Beta(a+#Obese, b+#NotObese)
    county = wdf.iloc[i,:]
    obesity2013 = float(county['Number2013'])
    percent2013 = float(county['Percent2013'])
    population2013 = float(county['Population2013'])
    samples[i,] = beta.rvs(a+obesity2013,b+population2013-obesity2013,size=samples.shape[1])


# find most obese county on each run
most_obese_runs = np.argmax(samples,axis=0)

# Find frequences of counties have been ranked most obese
counts = np.bincount(most_obese_runs)
# Find idex of most frequent county
most_obese_ranking = np.argmax(counts) 


print("Most Obese County:\n *** "+wdf.iloc[most_obese_ranking,:]['State']+','+ wdf.iloc[most_obese_ranking,:]['County']
      +" = ranked first "+str(counts[most_obese_ranking])+" times out of 1000 runs")

