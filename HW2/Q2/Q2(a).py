import numpy as np
import scipy.stats as stats
import pandas as pd
from Stat_function import t_test

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
pos_case = []               # number of positive cases for each county
num_samples_explore = 100   # number of samples to exploit each county
total_invitations = 10000   # total invitations
buy_rate = 0.2              # rate of buying when being invited
earn = 0                    # money earned

# randomized experiment - exploration: A/B testing
for i in range(0,df.shape[0]):
    case = 0
    sample = np.random.random_sample(num_samples_explore)
    total_invitations -= num_samples_explore
    earn -= 0.001 * num_samples_explore
    for j in range(num_samples_explore):
        if sample[j] < df.iloc[i,:]['percent']:
            case += 1
            buy = np.random.rand()
            if buy < buy_rate:
                earn += 2.99
    pos_case.append(case)
   
# randomized experiment - exploitation
max_expr = np.argmax(pos_case)
test = np.random.random_sample(total_invitations)
earn -= 0.001 * total_invitations
for j in range(0,total_invitations):
    if test[j] < df.iloc[max_expr,:]['percent']:
        buy = np.random.rand()
        if buy < buy_rate:
            earn += 2.99
            
# find county with highest rate in data
max_data = np.argmax(df['percent'])
            
# print result
print '%s, %s has highest rate in data, with rate %0.3f' %(df.iloc[max_data,:]['State']
            ,df.iloc[max_data,:]['County'],df.iloc[max_data,:]['percent'])
print '%s, %s is most promising in A/B testing, with rate %0.3f' %(df.iloc[max_expr,:]['State']
            ,df.iloc[max_expr,:]['County'],float(pos_case[max_expr])/num_samples_explore)
print "Total Revenue is {0}".format(earn)

# t-test
confidence_level = 1-0.05/(df.shape[0]-1) # Bonferroni correction
threshold = stats.norm.ppf(confidence_level)
diff = 0.01
for i in range(df.shape[0]):
    if i != max_expr:
        test_result = t_test(pos_case[max_expr], float(pos_case[max_expr])/num_samples_exploit
             , pos_case[i], float(pos_case[i])/num_samples_exploit, 0)
        print "{0},{1} VS {2},{3}:".format(df.iloc[max_expr,:]['County'],df.iloc[max_expr,:]['State']
            ,df.iloc[i,:]['County'],df.iloc[i,:]['State'])
        print "    t-score = {0}, threshold = {1}".format(test_result,threshold)