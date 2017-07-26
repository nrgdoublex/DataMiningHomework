import pandas as pd
import matplotlib.pylab as pl
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import CV as cv

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

    features_to_keep = list(df.columns.values)[1:499]

    target_var = np.array([1-float_or_str(i) for i in df['target_movie_Pulp_Fiction_1994']])

    df['intercept'] = 1.0

    # creates a clean numpy matrix for the regression
    data = df[features_to_keep].copy().as_matrix()

    # centers the data
    for i in range(0,len(data)):
        data[i,data[i,]>0] = data[i,data[i,]>0] - sum(data[i,data[i,]>0])/sum(data[i,]>0)

    # ADD INTERACTION TERMS HERE

    return (data,target_var)

#train_file = "movies_training.csv"
train_file = "movies_validation.csv"

#output file
out_file = "f1_cv.txt"
out_file_hd = open(out_file,'w')

regularization_strength = [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3]
reg_type = "l2"
reg_var = "reg strength"
kfold = 20

F1_scores_validation = []
F1_scores_train = []

#read data
data_whole, target_whole = data_processing(train_file)

#store F1 score
f1score_validation = np.zeros((len(regularization_strength),kfold))
f1score_train = np.zeros((len(regularization_strength),kfold))

reg_idx = 0
for reg_str in regularization_strength:
    
    #gen kfold indice
    indice = cv.KFold(data_whole, kfold)
    #shuffle data before training
    shuffled_data, shuffled_target = cv.shuffle(data_whole,target_whole)
    for kfold_idx in range(0,kfold):
        #print("(%f,%d)" %(reg_str,kfold_idx))
        valid_indice = np.arange(indice[kfold_idx],indice[kfold_idx+1])
        train_indice = np.delete(np.arange(0,shuffled_data.shape[0]),valid_indice)
        #[kira]:divide data
        data_validation = shuffled_data[valid_indice]
        target_validation = shuffled_target[valid_indice]

        data_train = shuffled_data[train_indice]
        target_train = shuffled_target[train_indice]
       
        # Describe classifier and regularization type
        logr = LogisticRegression(penalty=reg_type,C=reg_str,fit_intercept=False,class_weight="balanced",solver="liblinear")
        # Train model
        logr.fit(X=data_train, y=target_train)
        # Predicted probabilities of label +1
        #     0.5 is an arbitrary number
        y_pred_validation = (logr.predict_proba(data_validation)[:,1] > 0.5) + 0
        y_pred_train = (logr.predict_proba(data_train)[:,1] > 0.5) + 0

        f1score_validation[reg_idx][kfold_idx] = f1_score(target_validation, y_pred_validation)
        f1score_train[reg_idx][kfold_idx] = f1_score(target_train, y_pred_train)
    
    out_file_hd.write("reg_str = %.3f, train mean F1 = %f\n" %(reg_str,np.mean(f1score_train[reg_idx])))
    out_file_hd.write("reg_str = %.3f, valid mean F1 = %f\n" %(reg_str,np.mean(f1score_validation[reg_idx])))
    
    reg_idx += 1



