import math

import numpy as np
import pandas as pd

from scipy import stats

from scipy.stats import norm

APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QTRS_PER_YEAR = 4

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
QUARTERLY = 'quarterly'
YEARLY = 'yearly'

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    QUARTERLY: QTRS_PER_YEAR,
    YEARLY: 1
}











def ret(mu, x):
    return float(mu.T.dot(x).values)

def vol(cov, x):
    return float(np.sqrt(x.T.dot(cov).dot(x)).values)

def var(mu, cov, alpha, x):
    return float(-mu.T.dot(x).values - norm.ppf(alpha) * np.sqrt(x.T.dot(cov).dot(x)).values)

def cvar(mu, cov, alpha, x):
    return float(-mu.T.dot(x).values - norm.pdf(norm.ppf(alpha)) / alpha * np.sqrt(x.T.dot(cov).dot(x)).values)



