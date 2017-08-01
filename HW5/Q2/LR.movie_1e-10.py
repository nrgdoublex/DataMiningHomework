import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as pl
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def float_or_str(x):
	if isfloat(x):
		return (x)
	else:
		return (-1)


def percent_to_float(x):
	if isfloat(x):
		return (x/100)
	else:
		return float(x.strip('%'))/100

def add_noise(x):
	if not isinstance(x, string_types):
		return (x + np.random.normal(loc=0.0, scale=1e-3))
	else:
		return (x)


def data_processing(filename):
    # Read file (must be in UFT-8 if using python version >= 3)
    df = pd.read_csv(filename)

    features_to_keep = list(df.columns.values)[1:566]

    target_var = np.array([1-float_or_str(i) for i in df['target_movie_Pulp_Fiction_1994']])

    df['intercept'] = 1.0

    # creates a clean numpy matrix for the regression
    data = df[features_to_keep].copy().as_matrix()

    # centers the data
    for i in range(0,len(data)):
        data[i,data[i,]>0] = data[i,data[i,]>0] - sum(data[i,data[i,]>0])/sum(data[i,]>0)

    # ADD INTERACTION TERMS HERE

    return (data,target_var)

train_sizes = [50, 100, 500,1000,2000, 5000, 10000, 25000]
train_file = "movies_training_aug.csv"
# YOU NEED TO CREATE THE VALIDATION FILE FIRST BEFORE RUNNING THE CODE
validation_file = "movies_validation.csv"

regularization_strength = 1e-10
reg_type = "l2"
reg_var = "reg strength"

F1_scores_validation = []
F1_scores_train = []

data_train, target_train = data_processing(train_file)
data_validation, target_validation = data_processing(validation_file)

for train_size in train_sizes:
    
    f1score_validation = 0
    f1score_train = 0
    
    for avg_run in range(0,20):

        dt_index = np.random.choice(len(target_train), size=train_size, replace=True)
        data_train_1 = data_train[dt_index,]
        target_train_1 = target_train[dt_index]
    
        print("Train size",train_size," run ",avg_run)

        # Describe classifier and regularization type
        logr = LogisticRegression(penalty=reg_type,C=regularization_strength,fit_intercept=False,class_weight="balanced",solver="liblinear")
        # Train model
        logr.fit(X=data_train_1, y=target_train_1)
        # Predicted probabilities of label +1
        #     0.5 is an arbitrary number
        y_pred_validation = (logr.predict_proba(data_validation)[:,1] > 0.5) + 0
        y_pred_train_1 = (logr.predict_proba(data_train_1)[:,1] > 0.5) + 0

        f1score_validation += f1_score(target_validation, y_pred_validation)/20.

        f1score_train += f1_score(target_train_1, y_pred_train_1)/20.
        
    F1_scores_validation.append(f1score_validation)
    F1_scores_train.append(f1score_train)

#prepare plots
fig, ax = pl.subplots()
print(F1_scores_validation)
pl.plot(train_sizes, F1_scores_train, label='F1 over {type_val}, {type_pen} penalty ${reg_var} = {s}$'.format(type_val="training",type_pen=reg_type.upper(),reg_var=reg_var,s=regularization_strength),color="blue")
pl.plot(train_sizes, F1_scores_validation, label='F1 over {type_val}, {type_pen} penalty ${reg_var} = {s}$'.format(type_val="validation",type_pen=reg_type.upper(),reg_var=reg_var,s=regularization_strength),color="red")
#ax.set_ylim([0.5,1.1])
pl.legend(loc='best')
pl.xlabel('Training Data Size')
pl.ylabel('F1 Score')
pl.title('F1 Score x Training Data Size')

pl.savefig("movie_reg1e-10.png")
