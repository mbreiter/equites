import pandas as pd
# import pyfolio as pf

from tiingo import get_data
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from empyrical import *

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

try:
    # fast versions
    import bottleneck as bn

    def _wrap_function(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            out = kwargs.pop('out', None)
            data = f(*args, **kwargs)
            if out is None:
                out = data
            else:
                out[()] = data

            return out

        return wrapped

    nanmean = _wrap_function(bn.nanmean)
    nanstd = _wrap_function(bn.nanstd)
    nansum = _wrap_function(bn.nansum)
    nanmax = _wrap_function(bn.nanmax)
    nanmin = _wrap_function(bn.nanmin)
    nanargmax = _wrap_function(bn.nanargmax)
    nanargmin = _wrap_function(bn.nanargmin)
except ImportError:
    # slower numpy
    nanmean = np.nanmean
    nanstd = np.nanstd
    nansum = np.nansum
    nanmax = np.nanmax
    nanmin = np.nanmin
    nanargmax = np.nanargmax
    nanargmin = np.nanargmin

portfolio = {'AAPL': 500, 'MSFT': 1010, 'JPM': 2005, 'NTIOF': 3000}
start_date = (datetime.now() - relativedelta(years=6)).strftime("%Y-%m-%d")

def back_test(portfolio, start_date, end_date=None):

    if end_date is None: end_date = datetime.now().strftime("%Y-%m-%d")
    dollars = sum(portfolio.values())

    weights = {}

    for ticker in portfolio.keys():
        weights[ticker] = portfolio[ticker] / dollars

    prices = get_data(portfolio.keys(), 'adjClose', start_date, end_date, save=False, fail_safe=False)

    if prices is None:
        msg = "ERROR: could not retrieve pricing data for one or more of the assets given."
        print("\n\n{}".format(msg))

        return None, False, msg

    msg = "SUCCESS: retrieved pricing data all assets given."
    print("\n\n{}".format(msg))
    print(prices.tail(10))

    # check if any prices are missing ... if so drop the row
    if prices.isnull().values.any():
        msg += "\nWARNING: there are %f missing price entries ... truncating to a common base."
        print("\n\n{}".format(msg))

        prices.dropna(inplace=True)

    # check if the portfolio's budget is under utilized
    budget = sum(weights.values())

    print("\n\n{}".format(msg))

    # get the number of shares
    shares = pd.Series(portfolio) / prices.iloc[0]

    # calculate portfolio value per share ... to recover portfolio value per day, do value.sum(axis=1)
    value = prices * shares

    return value, True, msg


def annual_return(returns, period=DAILY, annualization=None):
    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)
    num_years = len(returns) / ann_factor
    # Pass array to ensure index -1 looks up successfully.
    ending_value = cum_returns_final(returns, starting_value=1)

    return ending_value ** (1 / num_years) - 1


def annualization_factor(period, annualization):
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor

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

def annual_volatility(returns,
                      period=DAILY,
                      alpha=2.0,
                      annualization=None,
                      out=None):
    
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    ann_factor = annualization_factor(period, annualization)
    nanstd(returns, ddof=1, axis=0, out=out)
    out = np.multiply(out, ann_factor ** (1.0 / alpha), out=out)
    if returns_1d:
        out = out.item()
    return out

def calmar_ratio(returns, period=DAILY, annualization=None):
    max_dd = max_drawdown(returns=returns)
    if max_dd < 0:
        temp = annual_return(
            returns=returns,
            period=period,
            annualization=annualization
        ) / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp

results = back_test(portfolio, start_date, end_date=None)

returns = results[0].sum(axis=1)
returns = (returns / returns.shift(1) - 1).dropna()
# pf.create_full_tear_sheet(returns)

Y=annual_return(returns)
M=max_drawdown(returns)

