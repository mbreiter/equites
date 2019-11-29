import os
import pandas as pd
import torch

from datetime import datetime
from scipy.stats.mstats import gmean
from dateutil.relativedelta import relativedelta

from stats import *
from cost import costs
from risk import risk_prefs
from tiingo import get_data
from optimize import optimize
from bl import bl
from rs import fama_french, regime_switch, current_regime, business_days, expected_returns, covariance
from sent_analysis import SentimentAnalysis, predict


def do_regime_switch(R, F, tickers):
    return regime_switch(R, F, tickers)



def get_estimates(tickers, prices, factors, mktcap, days, rs=True):
    returns = (prices / prices.shift(1) - 1).dropna()

    factors = factors[-(len(factors)-1):]

    R = returns.values
    F = np.hstack((np.atleast_2d(np.ones(factors.shape[0])).T, factors))

    if rs:
        transmat, loadings, covariances = do_regime_switch(R, F, tickers)

        baseline = 30
        regime = current_regime(R, F, loadings, baseline)
        print('\nbased on the last %d trading days, the best fitted regime is %d' % (baseline, regime))

        mu_factor = pd.DataFrame(days * expected_returns(F, transmat, loadings, regime), index=tickers, columns=['returns'])
        cov_factor = pd.DataFrame(days * covariance(R, F, transmat, loadings, covariances, regime), index=tickers, columns=tickers)
    else:
        beta = np.linalg.inv(F.T.dot(F)).dot(F.T).dot(R)

        mu_factor = pd.DataFrame(days * beta.T.dot(F.mean(axis=0)), index=tickers, columns=['returns'])
        cov_factor = pd.DataFrame(days * beta.T.dot(pd.DataFrame(F).cov().values).dot(beta) + pd.DataFrame(R - F.dot(beta)).cov().values, index=tickers, columns=tickers)

    l = (gmean(factors.iloc[-days:, :]['MKT'] + 1, axis=0) - 1) / factors.iloc[-days:, :]['MKT'].var()

    mu_bl1, cov_bl1 = bl(tickers=tickers,
                         l=l, tau=1,
                         mktcap=mktcap,
                         Sigma=returns.iloc[-days:, :].cov().values * days,
                         P=np.identity(len(tickers)),
                         Omega=np.diag(np.diag(cov_factor)),
                         q=mu_factor.values,
                         adjust=False)

    # mu_ml = mu_bl1.mul(pd.DataFrame(1 + np.random.uniform(-0.05, 0.1, len(tickers)), index=mu_bl1.index, columns=mu_bl1.columns))

    df = pd.DataFrame()
    df['prices'] = prices.apply(lambda x: ','.join(x.astype(str)), axis=1)
    df['prices'] = df.prices.apply(lambda x: [float(y) for y in x.split(',')])
    df.prices = df.prices.apply(lambda x: ast.literal_eval(x))
    mu_ml = predict(torch.FloatTensor(df['prices'].values.tolist(), check_ml=mu_bl1)

    mu_bl2, cov_bl2 = bl(tickers=tickers,
                         l=l, tau=1,
                         mktcap=mktcap,
                         Sigma=returns.iloc[-days:, :].cov().values * days,
                         P=np.identity(len(tickers)),
                         Omega=np.diag(np.diag(cov_factor)),
                         q=mu_ml.values,
                         adjust=True)

    return (mu_bl1, mu_bl2), (cov_bl1*100, cov_bl2*100), l
