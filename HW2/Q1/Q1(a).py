# -*- coding: latin-1 -*-

import numpy as np
import scipy.stats as stats
import pandas as pd

datafile = "Q1.csv"
columnname = {'number': 'Number2013', 'percent': 'Percent2013'
           , 'number.1': 'Number2012', 'percent.1': 'Percent2012'
           , 'number.2': 'Number2004', 'percent.2': 'Percent2004'}

# read csv and clean data
df = pd.read_csv(datafile, encoding='latin1', header=1)
df.rename(columns=columnname, inplace=True)
for column in ['Number2013','Percent2013','Number2012','Percent2012','Number2004','Percent2004']:
    df[column] = df[column].apply(pd.to_numeric, errors='coerce')
df['Percent2013'] = df['Percent2013'].div(100)
df['Percent2012'] = df['Percent2012'].div(100)
df['Percent2004'] = df['Percent2004'].div(100)
df['TotalPopulation2013'] = df.Number2013.div(df.Percent2013).apply(lambda x: int(x) if pd.notnull(x) else x)
df['TotalPopulation2012'] = df.Number2012.div(df.Percent2012).apply(lambda x: int(x) if pd.notnull(x) else x)

# t-test
totalpopu2013 = df['TotalPopulation2013'].sum()
totalpopu2012 = df['TotalPopulation2012'].sum()
obesitypopu2013 = df['Number2013'].sum()
obesitypopu2012 = df['Number2012'].sum()
totalper2013 = obesitypopu2013 / totalpopu2013
totalper2012 = obesitypopu2012 / totalpopu2012
se2013 = totalper2013 * (1 - totalper2013)
se2012 = totalper2012 * (1 - totalper2012)
probcombined = (obesitypopu2013 + obesitypopu2012) / (totalpopu2013 + totalpopu2012)
totalse = np.sqrt(se2013/totalpopu2013 + se2012/totalpopu2012)
t_score = (totalper2013 - totalper2012) / totalse

print 'The t-score for US population is %s' %t_score
print 'The t-score for 0.95 confidence is %s' %stats.norm.ppf(0.95)