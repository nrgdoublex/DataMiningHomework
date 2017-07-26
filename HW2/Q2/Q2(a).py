import numpy as np
from scipy.stats import binom

#constants
nodata = "No Data"

with open('Q2.csv') as f:
    lines = f.readlines()
    
#parameters
total_invi = 10000
m = 300
buy_rate = 0.2
    
# Data lists
state = []
county = []
num_2013 = []
per_2013 = []

for i in range(2,3226):
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

# get counties with more than 100,000 cases
county_idx = []
for i in range(0,len(num_2013)):
    if num_2013[i] > 100000:
        county_idx.append(i)
        #print '%s, %s' %(state[i],county[i])
        
#randomized experiment - exploration
pos_case = []
earn = 0
for i in range(0,len(county_idx)):
    case = 0
    sample = np.random.random_sample(m)
    total_invi = total_invi - m
    earn -= 0.001 * m
    for j in range(0,m):
        if sample[j] < per_2013[county_idx[i]]:
            case += 1
            buy = np.random.rand()
            if buy < buy_rate:
                earn += 2.99
    pos_case.append(case)
    
#randomized experiment - exploitation
max_idx = np.argmax(pos_case)
test = np.random.random_sample(total_invi)
earn -= 0.001 * total_invi
for j in range(0,total_invi):
    if test[j] < per_2013[county_idx[max_idx]]:
        buy = np.random.rand()
        if buy < buy_rate:
            earn += 2.99

#find county that achieves maximum probability
max_proba = 0
max_proba_idx = 0
second_proba = 0
second_proba_idx = 0
for i in range(0,len(county_idx)):
    if per_2013[county_idx[i]] > max_proba:
        second_proba = max_proba
        second_proba_idx = max_proba_idx
        max_proba = per_2013[county_idx[i]]
        max_proba_idx = i

print '%s,%s' %(state[county_idx[max_proba_idx]],county[county_idx[max_proba_idx]])
print max_proba
print '%s,%s' %(state[county_idx[second_proba_idx]],county[county_idx[second_proba_idx]])
print second_proba

for i in range(0,len(county_idx)):
    print '\item %s, %s \(\Longrightarrow\) %0.3f' %(state[county_idx[i]],county[county_idx[i]],float(pos_case[i])/m)

print pos_case
print '%s, %s => %0.3f' %(state[county_idx[max_idx]],county[county_idx[max_idx]],float(pos_case[max_idx])/m)
print earn