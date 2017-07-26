import numpy as np
import scipy.stats as st

diff = 0.001

#z-score function
def zscore(pos_1,p1,pos_2,p2,diff):
    popu_1 = pos_1/p1
    popu_2 = pos_2/p2
    total_proba = (pos_1+pos_2)/(popu_1+popu_2)
    se = np.sqrt(total_proba*(1-total_proba)*(1/popu_1+1/popu_2))
    z_score = ((p1 - p2)-diff)/se
    return z_score

#constants
nodata = "No Data"
scale = 10

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

#find index of Tippecanoe county 
tippecanoe = []    
for i in range(0,len(state)):
    if state[i] == 'Indiana' and county[i] =='Tippecanoe County':
        tippecanoe.append(i)
tippecanoe = tippecanoe[0]

for i in range(0,10):
    temp = i*diff
    z_score = zscore(num_2013[tippecanoe],per_2013[tippecanoe],num_2004[tippecanoe],per_2004[tippecanoe],temp)
    print temp
    print z_score
print 'The z-score for 0.995 confidence is %s' %st.norm.ppf(0.995)