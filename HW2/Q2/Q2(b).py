import numpy as np

#constants
nodata = "No Data"

with open('Q2.csv') as f:
    lines = f.readlines()
    
#parameters
total_invi = 10000
m = 300
buy_rate = 0.2
zscore = 2.862736
confidence_interval = 0.0015
    
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
        print '%s, %s => %f' %(state[i],county[i],per_2013[i])
        

#calculate sample size        
sample_size = np.zeros(len(county_idx))
for i in range(0,len(sample_size)):
    sample_size[i] = np.square(zscore)*per_2013[county_idx[i]]*(1-per_2013[county_idx[i]]) \
    /np.square(confidence_interval)
    print '\item %s, %s \(\Longrightarrow\) %d' %(state[county_idx[i]],county[county_idx[i]],np.ceil(sample_size[i]))
    
print np.max(np.ceil(sample_size))
