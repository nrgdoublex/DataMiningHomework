import numpy as np
import scipy.stats as st

#constants
nodata = "No Data"

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

#calculate US population
total_popu_2013 = 0
obe_popu_2013 = 0
totalper_2013 = float(0)

total_popu_2012 = 0
obe_popu_2012 = 0
totalper_2012 = float(0)
for i in range(0,len(num_2013)):
    if num_2013[i] == 0 or per_2013[i] == 0:
        continue
    else:   
        total_popu_2013 += num_2013[i]/per_2013[i]
        obe_popu_2013 += num_2013[i]
for i in range(0,len(num_2012)):
    if num_2012[i] == 0 or per_2012[i] == 0:
        continue
    else:   
        total_popu_2012 += num_2012[i]/per_2012[i]
        obe_popu_2012 += num_2012[i]
        
# calculate overall rate
totalper_2013 = float(np.sum(num_2013))/total_popu_2013
totalper_2012 = float(np.sum(num_2012))/total_popu_2012
total_proba = (obe_popu_2013+obe_popu_2012)/(total_popu_2013+total_popu_2012)
se = np.sqrt(total_proba*(1-total_proba)*(1/total_popu_2013+1/total_popu_2012))
z_score = (totalper_2013 - totalper_2012)/se

print 'The z-score for US population is %s' %z_score
print 'The z-score for 0.95 confidence is %s' %st.norm.ppf(0.95)

#print binom.sf(np.sum(num_2013)/10000,round(popu_2013/10000),totalper_2012)