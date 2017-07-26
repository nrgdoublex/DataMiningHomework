import numpy as np
import pandas as pd

def nonzero(a):
    if a != 0:
        return 1
    else:
        return 0

#df = pd.read_csv("movies_training_nor.csv")
 
#out = df.iloc[:1000,]

#out.to_csv("test.csv",index=False)
vecfunc = np.vectorize(nonzero)
a = [[1,2,3],[4,5,6]]
b = np.array([1,2])
#print(vecfunc(a))
print(1./b)


a = np.random.randint(low=-1,high=1,size=(3,3))
print(a)
print(np.sum(np.abs(a),axis = 1))
 
    