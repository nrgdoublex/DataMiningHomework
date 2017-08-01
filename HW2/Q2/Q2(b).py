import numpy as np
import scipy.stats as stats
import pandas as pd
from Stat_function import t_test

file_name = 'Q2.csv'
df = pd.read_csv(file_name
                 ,skiprows = 1
                 ,encoding = 'Latin-1'
                 ,na_values = 'No Data')
pd.set_option('display.expand_frame_repr', False)
df['number'] = df['number'].apply(lambda x: int(x) if pd.notnull(x) else x)
df['percent'] = df['percent'].div(100)

# we only care about those county with > 100000 people
df = df.loc[df['number'] >= 100000,:]
df.index = xrange(df.shape[0])

#t-score for 95% confidence interval(Bonferroni correction)
z = stats.norm.ppf(1-0.05/(df.shape[0]-1))
max = np.max(df['percent'])
argmax = np.argmax(df['percent'])
for i in range(df.shape[0]):
    if i == argmax: continue
    compare = np.max(df.iloc[i,:]['percent'])
    sample_size = (z**2)*max*(1-max)/((max-compare)**2)
    print "{0},{1} VS {2},{3} => sample size = {4}".format(
        df.iloc[argmax,:]['County'],df.iloc[argmax,:]['State'],
        df.iloc[i,:]['County'],df.iloc[i,:]['State'], sample_size)