import numpy as np
import scipy.stats as st

#constants
nodata = "No Data"
scale = 100
confidence_level = 0.05
threshold = st.norm.ppf(confidence_level)
diff = 0.00

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
        
print 'The county with largest empirical rate in 2004 is %s,%s, with rate %f' \
    %(state[np.argmax(per_2004)],county[np.argmax(per_2004)],np.max(per_2004))
print 'The county with largest empirical rate in 2012 is %s,%s, with rate %f' \
    %(state[np.argmax(per_2012)],county[np.argmax(per_2012)],np.max(per_2012))
print 'The county with largest empirical rate in 2013 is %s,%s, with rate %f' \
    %(state[np.argmax(per_2013)],county[np.argmax(per_2013)],np.max(per_2013))