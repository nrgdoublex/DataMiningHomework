# -*- coding: latin-1 -*-

import numpy as np
import scipy.stats as stats
import pandas as pd
from Statfunction import ttest
from Statfunction import degreefreedom


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

# compare 2004 and 2013 in Tippecanoe
tippecanoe = df.loc[(df['State'] == "Indiana") & (df['County'] == "Tippecanoe County")]
tippepopu2013 = float(tippecanoe.loc[:,'Number2013'])
tippeper2013 = float(tippecanoe.loc[:,'Percent2013'])
tippepopu2004 = float(tippecanoe.loc[:,'Number2004'])
tippeper2004 = float(tippecanoe.loc[:,'Percent2004'])
print "t-score of 2013 VS 2004 is %f" % ttest(tippepopu2013, tippeper2013, tippepopu2004, tippeper2004, 0)
print "Degree of Freedom of 2013 VS 2004 is %f" % degreefreedom(tippepopu2013, tippeper2013, tippepopu2004, tippeper2004)