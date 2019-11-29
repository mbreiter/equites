import pandas as pd
import numpy as np


from scipy.stats import t
from scipy.interpolate import griddata

def percentile(W, x):
    try:
        print(W)
        return np.percentile(W, x)
    except:
        return -1

def copula_pooling(mu, cov):

    # determine from matlab code ...
    df = 7

    # import the simulated returns from the copulas
    M = pd.read_csv('simulate_copulas.csv', header=None)

    # reorder the returns
    W = M.apply(lambda x: x.sort_values().values)

    # rank the returns
    C = M.rank(ascending=False) / (len(M))

    F_hat = W.apply(lambda x: t.cdf(x, df, mu[x.name], cov[x.name][x.name]))

    F = C.mul(F_hat) + (1-C) * C.mul(W.apply(lambda x: x.name, axis=1)/(len(W) + 1), axis=0)

    # the views
    V = pd.DataFrame(index=F.index)

    for j in range(len(mu)):
        V[j] = pd.DataFrame(W[j].quantile(F[j]).values)

    return V
