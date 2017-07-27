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

# Cook County VS Los Angeles Country
cook = df.loc[(df['State'] == "Illinois") & (df['County'] == "Cook County")]
LA = df.loc[(df['State'] == "California") & (df['County'] == "Los Angeles County")]
cookpopu2013 = float(cook.loc[:,'Number2013'])
cookper2013 = float(cook.loc[:,'Percent2013'])
LApopu2013 = float(LA.loc[:,'Number2013'])
LAper2013 = float(LA.loc[:,'Percent2013'])
print "t-score of Cook VS Los Angeles is %f" % ttest(cookpopu2013, cookper2013, LApopu2013, LAper2013, 0)
print "Degree of Freedom of Cook VS Los Angeles is %f" % degreefreedom(cookpopu2013, cookper2013, LApopu2013, LAper2013)

# Garfield County VS Ohio County
garfield = df.loc[(df['State'] == "Nebraska") & (df['County'] == "Garfield County")]
ohio = df.loc[(df['State'] == "Indiana") & (df['County'] == "Ohio County")]
garfieldpopu2013 = float(garfield.loc[:,'Number2013'])
garfieldper2013 = float(garfield.loc[:,'Percent2013'])
ohiopopu2013 = float(ohio.loc[:,'Number2013'])
ohioper2013 = float(ohio.loc[:,'Percent2013'])
print "t-score of Garfield VS Ohio is %f" % ttest(garfieldpopu2013, garfieldper2013, ohiopopu2013, ohioper2013, 0)
print "Degree of Freedom of Garfield VS Ohio is %f" % degreefreedom(garfieldpopu2013, garfieldper2013, ohiopopu2013, ohioper2013)