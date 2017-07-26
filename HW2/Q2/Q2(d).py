import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

#constants
nodata = "No Data"
    
#parameters
earn_money = 2.99
cost = 0.001
total_invi = 10000
m = 100
buy_rate = 0.2
epsilon = 0.2
total_case = []
pos_case = []
revenue = []
county_idx = []
round_idx = 0

    
# Data lists
state = []
county = []
num_2013 = []
per_2013 = []

# function to play arm
def play_arm(round,arm_idx,proba):
    global revenue
    global cost
    global buyrate
    global earn_money
    global county_idx
    global pos_case
    global total_case
    sample = np.random.rand()
    revenue[round] = revenue[round-1]
    revenue[round] -= cost
    # if we select positive case
    if sample < proba:
        pos_case[arm_idx][round] = pos_case[arm_idx][round-1] + 1
        buy = np.random.rand()
        # if he buys app
        if buy < buy_rate:
            revenue[round] += earn_money
    else:
        pos_case[arm_idx][round] = pos_case[arm_idx][round-1]
    #renew statistics of each round
    total_case[arm_idx][round] = total_case[arm_idx][round-1] + 1
    for i in range(0,len(county_idx)):
        if i != arm_idx:
            pos_case[i][round] = pos_case[i][round-1]
            total_case[i][round] = total_case[i][round-1]

def ucb(round_idx,len):
    global pos_case
    global total_case
    ucb = np.zeros(len)
    for i in range(0,len):
        if pos_case[:,round_idx][i] == 0 or round_idx == 1:
            ucb[i] = np.inf
        else:
            ucb[i] = float(pos_case[:,round_idx][i])/total_case[:,round_idx][i] + np.sqrt(2*np.log(round_idx-1)/total_case[:,round_idx][i])
    return np.argmax(ucb)  

#read data from file
with open('Q2.csv') as f:
    lines = f.readlines()
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
for i in range(0,len(num_2013)):
    if num_2013[i] > 100000:
        county_idx.append(i)
#        print '%s, %s' %(state[i],county[i])
        
#initialize parameters
round_idx = 0
pos_case = np.zeros((len(county_idx),total_invi+1))
total_case = np.zeros((len(county_idx),total_invi+1))
revenue = np.zeros(total_invi+1)
        
print pos_case[:,round_idx]
print total_case[:,round_idx]
#print np.argmax(pos_case[:,round_idx])
#-----------------------------------------UCB1 algorithm-----------------------------------------
# pull arm for rest of round
while round_idx < total_invi:
    best_arm_idx = ucb(round_idx,len(county_idx))
    round_idx += 1
    play_arm(round_idx,best_arm_idx, per_2013[county_idx[best_arm_idx]])
    
#print np.argmax(pos_case[:,round_idx])
#print pos_case[:,round_idx]
print 'The total revenue is %f' %revenue[round_idx]

# plot
dpi = 72
xinch = 1600/dpi
yinch = 800/dpi
fig = plt.figure(figsize=(xinch,yinch))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2]) 
#revenue plot
plot1 = plt.subplot(gs[0])
plot1.set_title('Total Revenue')
plot1.set_xlabel('number of total invitations')
plot1.set_ylabel('dollars')
plot1.plot(np.arange(total_invi+1),revenue,label='Revenue')
plot1.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=2, shadow=True, title="Legend", fancybox=True)

#obesity plot
plot2 = plt.subplot(gs[1])
plot2.set_title('Number of obesity cases in each selected county')
plot2.set_xlabel('number of total invitations')
plot2.set_ylabel('numbers of samples from a county')
for i in range(0,len(county_idx)):
    plot2.plot(np.arange(total_invi+1),total_case[i,:],label=county[county_idx[i]])
plot2.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=2, shadow=True, title="Legend", fancybox=True)
#save figure
plt.savefig('Q2(d).png')