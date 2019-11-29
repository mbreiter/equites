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

def max_drawdown(returns, out=None):
    """
    Determines the maximum drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    max_drawdown : float

    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns_array = np.asanyarray(returns)

    cumulative = np.empty(
        (returns.shape[0] + 1,) + returns.shape[1:],
        dtype='float64',
    )
    cumulative[0] = start = 100
    cum_returns(returns_array, starting_value=start, out=cumulative[1:])

    max_return = np.fmax.accumulate(cumulative, axis=0)

    nanmin((cumulative - max_return) / max_return, axis=0, out=out)
    if returns_1d:
        out = out.item()
    elif allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out











def ret(mu, x):
    return float(mu.T.dot(x).values)

def vol(cov, x):
    return float(np.sqrt(x.T.dot(cov).dot(x)).values)

def var(mu, cov, alpha, x):
    return float(-mu.T.dot(x).values - norm.ppf(alpha) * np.sqrt(x.T.dot(cov).dot(x)).values)

def cvar(mu, cov, alpha, x):
    return float(-mu.T.dot(x).values - norm.pdf(norm.ppf(alpha)) / alpha * np.sqrt(x.T.dot(cov).dot(x)).values)



