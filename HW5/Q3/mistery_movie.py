import pandas as pd
import matplotlib.pylab as pl
import numpy as np
from six import string_types
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

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

class MisteryClassifier:
    def __init__(self,nWL,tdp=1):
        self.wls = []
        self.weights = []
        self.nWL = nWL
        self.tdp = tdp
	
    def fit(self,X,y):
		# input: dataset X and labels y (in {+1, -1})
        N = X.shape[0]
        w = np.ones(N) / N

        for m in range(self.nWL):
            w_learn = DecisionTreeClassifier(max_depth=self.tdp,criterion="entropy")

            norm_w = np.exp(w)
            norm_w /= norm_w.sum()
            w_learn.fit(X, y, sample_weight=norm_w)
            yhat = w_learn.predict(X)

            eps = norm_w.dot(yhat != y) + 1e-20
            alpha = (np.log(1 - eps) - np.log(eps)) / 2

            w = w - alpha * y * yhat

            self.wls.append(w_learn)
            self.weights.append(alpha)

    def predict(self,X):
        y = np.zeros(X.shape[0])
        for (w_learn, alpha) in zip(self.wls, self.weights):
            y = y + alpha * w_learn.predict(X)
        return (np.sign(y))

# read the data in

def data_processing(filename):
    # Read file (must be in UFT-8 if using python version >= 3)
    df = pd.read_csv(filename)

    features_to_keep = list(df.columns.values)[1:499]

    target_var = np.array([1-2*float_or_str(i) for i in df['target_movie_Pulp_Fiction_1994']])

    df['intercept'] = 1.0

    # creates a clean numpy matrix for the regression
    data = df[features_to_keep].copy().as_matrix()

    # centers the data
    for i in range(0,len(data)):
        data[i,data[i,]>0] = data[i,data[i,]>0] - sum(data[i,data[i,]>0])/sum(data[i,]>0)

    # ADD INTERACTION TERMS HERE

    return (data,target_var)

train_sizes = [50, 100, 1000, 3000]
train_file = "movies_training.csv"
validation_file = "movies_validation.csv"

F1_scores_validation = []
F1_scores_train = []

data_train, target_train = data_processing(train_file)
data_validation, target_validation = data_processing(validation_file)

for train_size in train_sizes:
    
    f1score_validation = 0
    f1score_train = 0
    
    for avg_run in range(0,3):

        dt_index = np.random.choice(len(target_train), train_size)
        data_train_1 = data_train[dt_index,]
        target_train_1 = np.array(target_train)[dt_index]
        
        print("Train size",train_size," run ",avg_run)
    
        # Describe classifier and regularization type
        misteryCl = MisteryClassifier(nWL = 500, tdp=1)
        misteryCl.fit(X=data_train_1, y=target_train_1)
        # Predicted probabilities of label +1
        #     0.5 is an arbitrary number
        y_pred_validation = misteryCl.predict(data_validation)
        y_pred_train_1 = misteryCl.predict(data_train_1)

        f1score_validation += f1_score(target_validation, y_pred_validation)/3.

        f1score_train += f1_score(target_train_1, y_pred_train_1)/3.
        
    F1_scores_validation.append(f1score_validation)
    F1_scores_train.append(f1score_train)

#prepare plots
fig, ax = pl.subplots()
print(F1_scores_train)
print(F1_scores_validation)
pl.plot(train_sizes, F1_scores_train, label='F1 score over trainining',color="blue")
pl.plot(train_sizes, F1_scores_validation, label='F1 score over validation',color="red")
ax.set_ylim([0.1,1.1])
pl.legend(loc='best')
pl.xlabel('Training Data Size')
pl.ylabel('F1 Score')
pl.title('F1 Score x Training Data Size')

pl.savefig("test.png")
