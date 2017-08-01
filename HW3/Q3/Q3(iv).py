import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import scipy.stats as stats

def one_tailed_ttest(data1,data2):
    len_data1 = len(data1)
    len_data2 = len(data2)
    mean_data1 = np.average(data1)
    mean_data2 = np.average(data2)
    var_data1 = np.var(data1)
    var_data2 = np.var(data2)
    
    t_score = (mean_data1-mean_data2) / np.sqrt(var_data1/len_data1+var_data2/len_data2)
    df = np.square(var_data1/len_data1+var_data2/len_data2) / \
        (np.square(var_data1/len_data1)/(len_data1-1) + np.square(var_data2/len_data2)/(len_data2-1))
    p_value = stats.t.sf(t_score, df)
    
    return (t_score,p_value)

# training file
training_file = 'movies_training_IICF.csv'

# choose algorithm
algorithm_set = ['logistic','linear']

# parameters for LR
sigma_sq = 0.01
reg_type = 'l2'

# classification threshold
threshold = 0.4
# k-fold cross validation
k_fold = 10
# p-value threshold
p_value_thres = 0.05

# read training and validation data
df = pd.read_csv(training_file)
#storing accuracy
logistic_accuracy_validation = np.zeros(k_fold)
logistic_accuracy_train = np.zeros(k_fold)
linear_accuracy_validation = np.zeros(k_fold)
linear_accuracy_train = np.zeros(k_fold)

for idx, algorithm in enumerate(algorithm_set):
    # shuffle original DataFrame for future split
    df.index = np.random.permutation(df.index)
    df.sort_index(axis=0,inplace=True)
    
    # decide which part of k-fold
    chunk_size = df.shape[0] / k_fold
    for part in range(k_fold):
        indices = np.arange(0,df.shape[0])
        
        # split indices to decide which part to training and test
        training_indices = filter(lambda x: x / chunk_size  == part, indices)
        if k_fold * chunk_size < df.shape[0] and part < df.shape[0] - k_fold * chunk_size: 
            training_indices = training_indices + [indices[-part-1]]
        validation_indices = [index for index in indices if index not in training_indices]
        
        # split dataframe into training and test set
        df_train = df.iloc[training_indices,:].copy()
        df_valid = df.iloc[validation_indices,:].copy()
        
        target_train = df_train['target_movie_Pulp_Fiction_1994']
        data_train = df_train.iloc[:,1:(df_train.shape[1]-1)]
        target_validation = df_valid['target_movie_Pulp_Fiction_1994']
        data_validation = df_valid.iloc[:,1:(df_valid.shape[1]-1)]
        
        # build Model
        if algorithm =='logistic':
            model = LogisticRegression(penalty=reg_type,C=1/sigma_sq,fit_intercept=True)
        elif algorithm == 'linear':
            model = SGDClassifier(loss='squared_loss',penalty='none',fit_intercept=True,shuffle=False, \
                                  random_state=0,eta0=0.01,power_t=0.5)
        # fit model    
        model.fit(X=data_train, y=target_train)
        
        # prediction on validation set
        if algorithm =='logistic':
            y_pred_validation = (model.predict_proba(data_validation)[:,1] > threshold) + 0
        elif algorithm == 'linear':
            y_pred_validation = model.predict(data_validation)
            
        # f1 score
        f1score_validation = f1_score(target_validation, y_pred_validation)   
        
        # storing F1 score 
        if algorithm == 'logistic':
            logistic_accuracy_validation[part] = f1score_validation
        elif algorithm == 'linear':
            linear_accuracy_validation[part] = f1score_validation 
            
# plot distribution
data_logistic = sorted(logistic_accuracy_validation)
fit_logistic = stats.norm.pdf(data_logistic, np.mean(data_logistic), np.std(data_logistic))
pl.plot(data_logistic,fit_logistic,'-o',label='Logistic Regression')

data_linear = sorted(linear_accuracy_validation)
fit_linear = stats.norm.pdf(data_linear, np.mean(data_linear), np.std(data_linear))
pl.plot(data_linear,fit_linear,'-v',label='SGDClassifier')

plt.title('F1 score vs. threshold')
plt.xlabel('F1 score')
plt.ylabel('Frequency')
plt.legend(loc="upper left")
plt.savefig("Q3(iv).png")

(t_score,p_value) = one_tailed_ttest(logistic_accuracy_validation, linear_accuracy_validation)
print "The T-statistic of LR VS SGD = {0}".format(t_score)
print "The p-value of LR VS SGD = {0}".format(p_value)
if p_value < p_value_thres:
    print("Logistic Regression is more powerful")
else:
    print("We cannot differentiate the power of the two")