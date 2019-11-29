import os
import numpy as np
import pandas as pd
from regime_switch_model.rshmm import *

model = HMMRS(n_components=2)
# startprob
model.startprob_ = np.array([0.9, 0.1])

# transition matrix
model.transmat_ = np.array([[0.9, 0.1], [0.6, 0.4]])
# risk factor matrix
# read file from Fama-French three-factor data
Fama_French = pd.read_csv(os.getcwd() + r'/data/ff_factors_3.csv')
Fama_French.replace(-99.99, np.nan);
Fama_French.replace(-999, np.nan);

# select data
#Fama_French_subset = Fama_French[(Fama_French['TimeStamp'] >= 20150101) & (Fama_French['TimeStamp'] <= 20171231)]
Fama_French_subset = Fama_French
Fama_French_subset.drop(['Date', 'RF'], axis=1, inplace=True)
F = np.hstack((np.atleast_2d(np.ones(Fama_French_subset.shape[0])).T, Fama_French_subset))

# loading matrix with intercept
loadingmat1 = np.array([[0.9, 0.052, -0.02],
                        [0.3, 0.27, 0.01],
                        [0.12, 0.1, -0.05],
                        [0.04, 0.01, -0.15],
                        [0.15, 0.04, -0.11]])
intercept1 = np.atleast_2d(np.array([-0.015, -0.01, 0.005, 00.1, 0.02])).T

model.loadingmat_ = np.stack((np.hstack((intercept1, loadingmat1)),
                              np.hstack((0.25*intercept1, -0.5* loadingmat1))), axis=0)

# covariance matrix
n_stocks = 5
rho = 0.2
Sigma1 = np.full((n_stocks, n_stocks), rho) + np.diag(np.repeat(1-rho, n_stocks))
model.covmat_ = np.stack((Sigma1, 10*Sigma1), axis=0)

save = True
# sample
Y, Z = model.sample(F)


# Use the last 300 day as the test data
Y_train = Y[:-300,:]
Y_test = Y[-300:,:]
F_train = F[:-300,:]
F_test = F[-300:,:]

remodel = HMMRS(n_components=2, verbose=True)
remodel.fit(Y_train, F_train)
Z2, logl, viterbi_lattice = remodel.predict(Y_train, F_train)
