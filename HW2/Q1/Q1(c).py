import numpy as np
import scipy.stats as st

#z-score function
def zscore(pos_1,p1,pos_2,p2):
    popu_1 = pos_1/p1
    popu_2 = pos_2/p2
    total_proba = (pos_1+pos_2)/(popu_1+popu_2)
    se = np.sqrt(total_proba*(1-total_proba)*(1/popu_1+1/popu_2))
    z_score = (p1 - p2)/se
    return z_score

#constants
nodata = "No Data"
scale = 10000
scale2 = 10

with open('Q1.csv') as f:
    lines = f.readlines()

# Data lists
state = []
county = []
num_2013 = []
per_2013 = []
num_2012 = []
per_2012 = []
num_2004 = []
per_2004 = []
for i in range(2,3148):
    line = lines[i].rstrip().split(',')
    state.append(line[0])
    county.append(line[2])
    if line[3] == nodata:
        num_2013.append(0)
    else:
        num_2013.append(float(line[3]))
        
    if line[4]==nodata:
        per_2013.append(0)
    else:
        per_2013.append(float(line[4])/100)
                
    if line[5]==nodata:
        num_2012.append(0)
    else:
        num_2012.append(float(line[5]))
                
    if line[6]==nodata:
        per_2012.append(0)
    else:
        per_2012.append(float(line[6])/100)
                
    if line[7]==nodata:
        num_2004.append(0)
    else:
        num_2004.append(float(line[7]))
                
    if line[8]==nodata:
        per_2004.append(0)
    else:
        per_2004.append(float(line[8])/100)
        
#find index of counties
cook = []
la = []
garfield = []
ohio = []
for i in range(0,len(state)):
    if state[i] == 'Illinois' and county[i] =='Cook County':
        cook.append(i)
    elif (state[i] == 'California') and (county[i] =='Los Angeles County'):
        la.append(i)
    elif (state[i] == 'Nebraska') and (county[i] =='Garfield County'):
        garfield.append(i);
    elif (state[i] == 'Indiana') and (county[i] =='Ohio County'):
        ohio.append(i)
cook = cook[0]
la = la[0]
garfield = garfield[0]
ohio = ohio[0]

#find z-score between Cook County and LA county
print 'The z-score between Cook and LA in 2013 is %s' %zscore(num_2013[cook],per_2013[cook],num_2013[la],per_2013[la])
print 'The z-score between Cook and LA in 2012 is %s' %zscore(num_2012[cook],per_2012[cook],num_2012[la],per_2012[la])
print 'The z-score between Cook and LA in 2004 is %s' %zscore(num_2004[cook],per_2004[cook],num_2004[la],per_2004[la])

#find z-score between Garfield County and Ohio county
print 'The z-score between Garfield and Ohio in 2013 is %s' %zscore(num_2013[ohio],per_2013[ohio],num_2013[garfield],per_2013[garfield])
print 'The z-score between Garfield and Ohio in 2012 is %s' %zscore(num_2012[ohio],per_2012[ohio],num_2012[garfield],per_2012[garfield])
print 'The z-score between Garfield and Ohio in 2004 is %s' %zscore(num_2004[ohio],per_2004[ohio],num_2004[garfield],per_2004[garfield])

print 'The z-score for 0.95 confidence is %s' %st.norm.ppf(0.95)

