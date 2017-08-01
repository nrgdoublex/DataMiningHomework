import numpy as np
import scipy.stats as stats
import pandas as pd
from Stat_function import t_test
import matplotlib.pyplot as plt
from matplotlib import gridspec
from Sample import sample

file_name = 'Q2.csv'
df = pd.read_csv(file_name
                 ,skiprows = 1
                 ,encoding = 'Latin-1'
                 ,na_values = 'No Data')
pd.set_option('display.expand_frame_repr', False)
df['number'] = df['number'].apply(lambda x: int(x) if pd.notnull(x) else x)
df['percent'] = df['percent'].div(100)

# we only care about those county with > 100000 people
df = df.loc[df['number'] >= 100000,:]
df.index = xrange(df.shape[0])

# some parameters to do MAB
pos_case = np.zeros(df.shape[0])               # number of positive cases for each county
num_samples_explore = 100   # number of samples to exploit each county
total_invitations = 10000   # total invitations
buy_rate = 0.2              # rate of buying when being invited
earn = 0                    # money earned
epsilon = 0.1               # epsilon for epsilon-greedy sampling
revenue_array = []
num_invitations = 0
num_invitations_county = np.zeros((df.shape[0],total_invitations))
round_idx = 0

#randomized experiment - to get best arm initially
for explore_idx in range(num_samples_explore):
    for county in range(0,df.shape[0]):
        earn -= 0.001
        positive, buy = sample(df.iloc[county,:]['percent'],buy_rate)
        
        # calculate number of invitations VS number of pulling arms
        if positive: 
            pos_case[county] += 1
            num_invitations += 1
        for i in range(df.shape[0]):
            num_invitations_county[i][round_idx] = pos_case[i]
        
        # calculate revenue VS number of number of invitations
        if buy:
            earn += 2.99
        if positive: revenue_array.append(earn)
        
    round_idx += 1
    
#randomized experiment - epsilon-greedy sampling
best_arm = np.argmax(pos_case)
# pull arm for rest of round

while round_idx < total_invitations:
    #decide to pull best arm or other arms
    rv = np.random.rand()
    earn -= 0.001
    
    # choose arm to pull
    if rv >= epsilon:
        arm = best_arm
    else:
        #play random arms
        arm = np.random.randint(0,len(pos_case))
        
    # play arm
    positive, buy = sample(df.iloc[arm,:]['percent'],buy_rate)
    
    # calculate number of invitations VS number of pulling arms
    if positive: 
        pos_case[best_arm] += 1
        num_invitations += 1
    for i in range(df.shape[0]):
        num_invitations_county[i][round_idx] = pos_case[i]
    
    # calculate revenue VS number of number of invitations
    if buy:
        earn += 2.99
    if positive: revenue_array.append(earn)
    
    #decide the best arm so far
    best_arm = np.argmax(pos_case)

    round_idx += 1
    
# test
# print len(revenue_array)
# print num_invitations_county.shape
    
# plot
dpi = 72
xinch = 1600/dpi
yinch = 800/dpi
fig = plt.figure(figsize=(xinch,yinch))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2]) 
#revenue plot
plot1 = plt.subplot(gs[0])
plot1.set_title('Total Revenue')
plot1.set_xlabel('number of invitations')
plot1.set_ylabel('dollars')
plot1.plot(np.arange(1,len(revenue_array)+1),revenue_array,label='Revenue')
plot1.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=2, shadow=True, title="Legend", fancybox=True)

#obesity plot
plot2 = plt.subplot(gs[1])
plot2.set_xlabel('number of samples')
plot2.set_ylabel('numbers of invitations from a county')
for i in range(df.shape[0]):
    plot2.plot(np.arange(1,total_invitations+1),
               num_invitations_county[i],
               label=df.iloc[i]['County']+','+df.iloc[i]['State'])
plot2.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=2, shadow=True, title="Legend", fancybox=True)
#save figure
plt.savefig('Q2(c).png')

print "Total Revenue = %f" % earn