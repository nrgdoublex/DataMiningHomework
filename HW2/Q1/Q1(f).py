# -*- coding: latin-1 -*-

import numpy as np
import scipy.stats as stats
import pandas as pd
from Statfunction import ttest


datafile = "Q1.csv"
columnname = {'number': 'Number2013', 'percent': 'Percent2013'
           , 'number.1': 'Number2012', 'percent.1': 'Percent2012'
           , 'number.2': 'Number2004', 'percent.2': 'Percent2004'}

confidence_level = 1-0.05/3142
threshold = stats.norm.ppf(confidence_level)
diff = 0.01

# read csv and clean data
df = pd.read_csv(datafile, encoding='latin1', header=1)
df.rename(columns=columnname, inplace=True)
for column in ['Number2013','Percent2013','Number2012','Percent2012','Number2004','Percent2004']:
    df[column] = df[column].apply(pd.to_numeric, errors='coerce')
df['Percent2013'] = df['Percent2013'].div(100)
df['Percent2012'] = df['Percent2012'].div(100)
df['Percent2004'] = df['Percent2004'].div(100)

# do hypothesis test in each county to see which is significant
# use Bonferronicorrection
count = 0
with open('Q1(f)_ans.txt','w') as f:
    for _, county in df.iterrows():
        data = county[['State','County','Number2013','Percent2013','Number2004','Percent2004']]
        if any(pd.isnull(data)): continue
        popu2013 = float(data['Number2013'])
        per2013 = float(data['Percent2013'])
        popu2004 = float(data['Number2004'])
        per2004 = float(data['Percent2004'])
        
        t_score = ttest(popu2004, per2004, popu2013, per2013, diff)
        if t_score > threshold:
            f.write('%s, %s\n' %(data['State'], data['County']))
            print data['State'], data['County']
            count += 1
            
print count