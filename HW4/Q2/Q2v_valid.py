import operator
import numpy as np
from functools import reduce
import matplotlib.pyplot as pl
import sklearn.metrics


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

user_num = 29999
Xlist = []

maxi = 0
maxj = 0
maxk = 0
with open("tensor_train.csv") as f:
  for line in f.readlines():
    l = line.strip().split(",")
    i,j,k = map(int,l[0:3])
    Xijk = float(l[3])
    #i,j,k,Xijk = list(map(float, line.strip().split(",")))
    maxi = max(i,maxi)
    maxj = max(j,maxj)
    maxk = max(k,maxk)
    Xlist.append((i,j,k,Xijk))

X_ref = np.zeros((maxi+1, maxj+1, maxk+1))
for i,j,k,Xijk in Xlist:
    X_ref[i,j,k] = Xijk

#log
output = open("Q2_v_valid.txt",'w')

fold = 10
rank_set = np.arange(1,11)
#validation
for n in rank_set:
    print("rank number = %d" %n)
    #validation_user = [np.random.randint(1,user_num) for r in np.arange(user_num/fold)]
    X = np.zeros((maxi+1, maxj+1, maxk+1))
    for i,j,k,Xijk in Xlist:
        if i < 3000 and j == 500:
            continue
        else:
            X[i,j,k] = Xijk

    #PARAFAC algorithm
    loadings,mse = parafac_base(x=X,nfactors=n,max_iter=20)

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
    
    predict = [1 if i > 0 else 0 for i in rating[:3000]]
    target = [1 if i > 0 else 0 for i in X_ref[:3000,pulp_idx,pulp_genre_idx[0]]]
    
    #print("F1 score for rank %d=%f\n" %(n,sklearn.metrics.f1_score(target,predict)))
    output.write("F1 score for rank %d=%f\n" %(n,sklearn.metrics.f1_score(target,predict)))
