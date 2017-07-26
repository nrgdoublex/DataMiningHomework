import operator
import numpy as np
from functools import reduce
import matplotlib.pyplot as pl
import sklearn.metrics
import pandas as pd


# PARAFAC decomposition. Parts of this code come from http://stackoverflow.com/questions/36541154/how-to-use-scikit-tensor-in-python
def ribs(loadings):
  '''
  Convert a list of n loading matrices [A_{fi}, B_{fj}, C_{fk}, ...] into ribs
  [A_{fi11...}, B_{f1j1...}, C_{f11k...}, ...]. These ribs can be multiplied
  with numpy broadcasting to fill a tensor with data.
  '''
  loadings = [np.atleast_2d(l) for l in loadings]
  nfactors = loadings[0].shape[0]
  assert np.alltrue([l.ndim == 2 and l.shape[0] == nfactors for l in loadings])
  ribs = []
  for mi in range(len(loadings)):
    shape = [nfactors] + [-1 if fi == mi else 1 for fi in range(len(loadings))]
    ribs.append(loadings[mi].reshape(shape))
  return ribs

def para_compose(ribs):
  return np.sum(reduce(operator.mul, ribs), axis=0)

def parafac_base(x, nfactors, max_iter=100):
  '''
  PARAFAC is a multi-way tensor decomposition method. Given a tensor X, and a
  number of factors nfactors, PARAFAC decomposes the X in n factors for each
  dimension in X using alternating least squares:

  X_{ijk} = \sum_{f} a_{fi} b_{fj} c_{fk} + e_{ijk}

  PARAFAC can be seen as a generalization of PCA to higher order arrays [1].
  Return a ([a, b, c, ...], mse)

  [1] Rasmus Bro. PARAFAC. Tutorial and applications. Chemometrics and
  Intelligent Laboratory Systems, 38(2):149-171, 1997.
  '''
  loadings = [np.random.rand(nfactors, n) for n in x.shape]

  last_mse = np.inf
  for i in range(max_iter):
    # 1) forward (predict x)
    xhat = para_compose(ribs(loadings))

    # 2) stopping?
    mse = np.mean((xhat - x) ** 2)
    if last_mse - mse < 1e-10 or mse < 1e-20:
      break
    last_mse = mse

    for mode in range(len(loadings)):
      print('iter: %d, dir: %d' % (i, mode))
      # a) Re-compose using other factors
      Z = ribs([l for li, l in enumerate(loadings) if li != mode])
      Z = reduce(operator.mul, Z)

      # b) Isolate mode
      Z = Z.reshape(nfactors, -1).T # Z = [long x fact]
      Y = np.rollaxis(x, mode)
      Y = Y.reshape(Y.shape[0], -1).T # Y = [mode x long]

      # c) least squares estimation: x = np.lstsq(Z, Y) -> Z x = Y
      new_fact, _, _, _ = np.linalg.lstsq(Z, Y)
      loadings[mode] = new_fact
  if not i < max_iter - 1:
    print('parafac did not converge in %d iterations (mse=%.2g)' %
      (max_iter, mse))
  return loadings, mse

num_rank = 8
train_num = 30000
Xlist = []

maxi = 0
maxj = 0
maxk = 0
with open("tensor_train.csv") as f:
  for line in f.readlines():
    l = line.strip().split(",")
    i,j,k = map(int,l[0:3])
    Xijk = float(l[3])
    maxi = max(i,maxi)
    maxj = max(j,maxj)
    maxk = max(k,maxk)
    Xlist.append((i,j,k,Xijk))
with open("tensor_test.csv") as f:
  for line in f.readlines():
    l = line.strip().split(",")
    i,j,k = map(int,l[0:3])
    Xijk = float(l[3])
    maxi = max(i,maxi)
    maxj = max(j,maxj)
    maxk = max(k,maxk)
    Xlist.append((i,j,k,Xijk))


X = np.zeros((maxi+1, maxj+1, maxk+1))
for i,j,k,Xijk in Xlist:
    X[i,j,k] = Xijk

#PARAFAC algorithm
loadings,mse = parafac_base(x=X,nfactors=num_rank,max_iter=100)

user=loadings[0]
movie=loadings[1]
genre=loadings[2]

pulp_idx = 500
pulp_genre_idx = [4,7,9]
rating = np.zeros(user.shape[1])
for i in range(0,user.shape[0]):
    movie_coe = movie[i][pulp_idx]
    for j in range(0,len(pulp_genre_idx)):
        genre_coe = genre[i][pulp_genre_idx[j]]
        rating += user[i] * movie_coe * genre_coe
rating = rating / 3    

predict = [1 if i > 0 else 0 for i in rating[train_num:]]

df_test = pd.read_csv("movies_test_features_nor.csv")
df_output = pd.DataFrame(index=np.arange(len(df_test)),columns=['user_id','target_movie_Pulp_Fiction_1994'])
df_output['user_id'] = df_test['user_id']
df_output['target_movie_Pulp_Fiction_1994'] = predict
df_output.to_csv("Q2v_predict.csv",index=False)

