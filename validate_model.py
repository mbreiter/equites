import os

import matplotlib
import numpy as np
import pandas as pd

from bl import get_mkt_cap
from cost import costs
from risk import risk_prefs
from tiingo import get_data
from optimize import optimize
from estimates import get_estimates

from datetime import datetime
from dateutil.relativedelta import relativedelta



def business_days(start_date, end_date):
    return len(pd.bdate_range(start_date, end_date))

def fama_french(start_date, end_date, five):

    if five:
        factors = pd.read_csv(os.getcwd() + r'/data/ff_factors.csv', index_col=0)
    else:
        factors = pd.read_csv(os.getcwd() + r'/data/ff_factors_3.csv', index_col=0)

    factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)

    factors.replace(-99.99, np.nan)
    factors.replace(-999, np.nan)

    factors = factors / 100

    factors = factors.loc[start_date:end_date,:].iloc[1:]
    factors.index = pd.to_datetime(factors.index)

    return factors


def portfolio_value(weights, prices):
    return float(weights.values.T.dot(prices.values))


def do_optimize(mu, cov, alpha, return_target, cost, prices, risk_tolerance):
    soln = optimize(mu=(mu[0].values.ravel(), mu[1].values.ravel()),
                         sigma=(cov[0].values, cov[1].values),
                         alpha=(alpha, alpha * 1.02),
                         return_target=(return_target, return_target),
                         costs=cost,
                         prices=prices.iloc[-2, :].values if prices.iloc[-1, :].isnull().values.any() else prices.iloc[-1, :].values,
                         gamma=risk_tolerance)[0]

    if risk_tolerance[3] == "PARITY":
        x1 = soln.x[:int(len(mu[0]))]
        return pd.Series(x1/x1.sum(), index=mu[0].index)
    else:
        return pd.Series(soln.x[:int(len(mu[0]))], index=mu[0].index)


## *********************************************************************************************************************
# parameters
## *********************************************************************************************************************

# get the list of assets
tickers = list(pd.read_csv(os.getcwd() + r'/data/tickers.csv')['Tickers'])
N = len(tickers)

backtest_start = (datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d")
sample_start = (datetime.now() - relativedelta(years=9)).strftime("%Y-%m-%d")

simulation_end = datetime.now().strftime("%Y-%m-%d")

rs = False

## *********************************************************************************************************************
# get price data and factors
## *********************************************************************************************************************

# get market capitalizations
# mktcap = get_mkt_cap(tickers, save=True)
mktcap = pd.read_csv(os.getcwd() + r'/data/mkt_cap.csv', index_col=0)

# prices
# prices = get_data(tickers, 'adjClose', sample_start, simulation_end, save=True, fail_safe=True)
prices = pd.read_csv(os.getcwd() + r'/data/VALIDATE_PRICES.csv', index_col=0)
prices.to_csv(os.getcwd() + r'/data/VALIDATE_PRICES.csv')

# average volumes
# volumes = get_data(tickers, 'volume', sample_start, simulation_end, save=True)
volumes = pd.read_csv(os.getcwd() + r'/data/VALIDATE_VOLUMES.csv', index_col=0)
volumes.to_csv(os.getcwd() + r'/data/VALIDATE_VOLUMES.csv')

# factors
f5 = fama_french(sample_start, simulation_end, five=True)
f3 = fama_french(sample_start, simulation_end, five=False)
capm = f3['MKT']

f3.drop('RF', axis=1, inplace=True)
f5.drop('RF', axis=1, inplace=True)

# returns
returns = (prices / prices.shift(1) - 1).dropna()[:len(f5)].values

## *********************************************************************************************************************
#  risk preferences
## *********************************************************************************************************************

# in years
horizon = 10

# set the cardinality constraints
cardinality = []

# target goal amount in dollars
target_dollars, initial_dollars = 1100, 1000

# low 1, medium 2, high = 3
aversion = 1

## *********************************************************************************************************************
# calibrate simulation dates
## *********************************************************************************************************************

sample_start = 0
sample_length = business_days(f5.index[sample_start], datetime.strptime(backtest_start, "%Y-%m-%d"))
sample_end = sample_start + sample_length

sample_factors = f5[sample_start:sample_end]
sample_prices = prices[sample_start:sample_end]
sample_volumes = volumes[sample_start:sample_end]

test_start = sample_end
backtest_start_index = test_start

done = False

## *********************************************************************************************************************
# initialize tracking dataframes
## *********************************************************************************************************************
portfolio_values = pd.DataFrame([[1, 1, 1, 1]], index=[prices.index[test_start]], columns=['MCVAR', 'SHARPE', 'MVO', 'N', 'PARITY'])

MCVAR_holdings = pd.DataFrame(columns=tickers)
SHARPE_holdings = pd.DataFrame(columns=tickers)
MVO_holdings = pd.DataFrame(columns=tickers)
N_holdings = pd.DataFrame(columns=tickers)
PARITY_holdings = pd.DataFrame(columns=tickers)

validate = True

while test_start < len(f5) and not done and validate:
    # get the number of days in the testing period ... tests last 6 months
    test_start_date = f5.index[test_start]
    test_days = business_days(f5.index[test_start], f5.index[test_start] + relativedelta(months=6))
    test_end = test_start + test_days

    # test right up until we are done
    if test_end >= len(f5):
        test_end = len(f5) - 1
        done = True

    print("SAMPLE PERIOD: \t {} \t {}".format(f5.index[sample_start], f5.index[sample_end]))
    print("TEST PERIOD: \t {} \t {}".format(f5.index[test_start], f5.index[test_end]))

    # get estimates
    mu, cov, l = get_estimates(tickers, sample_prices, sample_factors, mktcap, test_days, rs=rs)

    # get the prices over the testing period
    test_prices = prices[test_start:test_end]

    # get cost estimates
    cost = costs(tickers=tickers,
                 cov=cov[0],
                 volumes=sample_volumes.mean(),
                 prices=test_prices.iloc[0,:],
                 start_date=f5.index[sample_start],
                 end_date=f5.index[sample_end],
                 alpha=5)

    # optimization parameters
    alpha, return_target, multipliers, exposures, cardinality = risk_prefs(horizon, aversion, initial_dollars,
                                                                           target_dollars, l, mu[0], mu[1], cov[0])

    # MCVAR optimization
    risk_tolerance = (multipliers, exposures, cardinality, 'MCVAR')
    MCVAR_portfolio = do_optimize(mu, cov, alpha, return_target, cost, prices, risk_tolerance)
    MCVAR_shares = (portfolio_values['MCVAR'].iloc[-1] / test_prices.iloc[0]).multiply(MCVAR_portfolio)
    MCVAR_values = pd.DataFrame((test_prices * MCVAR_shares).sum(axis=1), columns=["MCVAR"])
    MCVAR_holdings = MCVAR_holdings.append(pd.DataFrame(np.tile(MCVAR_shares,
                                                                (len(test_prices.index), 1)),
                                                        index=test_prices.index,
                                                        columns=tickers))


    # Sharpe optimization
    risk_tolerance = (multipliers, exposures, cardinality, 'SHARPE')
    SHARPE_portfolio = do_optimize(mu, cov, alpha, return_target, cost, prices, risk_tolerance)
    SHARPE_shares = (portfolio_values['SHARPE'].iloc[-1] / test_prices.iloc[0]).multiply(SHARPE_portfolio)
    SHARPE_values = pd.DataFrame((test_prices * SHARPE_shares).sum(axis=1), columns=["SHARPE"])
    SHARPE_holdings = SHARPE_holdings.append(pd.DataFrame(np.tile(SHARPE_shares,
                                                                (len(test_prices.index), 1)),
                                                        index=test_prices.index,
                                                        columns=tickers))

    # MVO optimization
    risk_tolerance = (multipliers, exposures, cardinality, 'MVO')
    MVO_portfolio = do_optimize(mu, cov, alpha, return_target, cost, prices, risk_tolerance)
    MVO_shares = (portfolio_values['MVO'].iloc[-1] / test_prices.iloc[0]).multiply(MVO_portfolio)
    MVO_values = pd.DataFrame((test_prices * MVO_shares).sum(axis=1), columns=['MVO'])
    MVO_holdings = MVO_holdings.append(pd.DataFrame(np.tile(MVO_shares,
                                                            (len(test_prices.index), 1)),
                                                          index=test_prices.index,
                                                          columns=tickers))

    # Equal weight contribution
    N_portfolio = pd.Series([1/N] * N, index=tickers)
    N_shares = (portfolio_values['N'].iloc[-1] / test_prices.iloc[0]).multiply(N_portfolio)
    N_values = pd.DataFrame((test_prices * N_shares).sum(axis=1), columns=["N"])
    N_holdings = N_holdings.append(pd.DataFrame(np.tile(N_shares,
                                                        (len(test_prices.index), 1)),
                                                    index=test_prices.index,
                                                    columns=tickers))

    # RISK PARITY Equal RISK PARITY  contribution
    risk_tolerance = (multipliers, (0, exposures[1]), cardinality, 'PARITY')
    RP_portfolio = do_optimize(mu, cov, alpha, return_target, cost, prices, risk_tolerance)
    RP_shares = (portfolio_values['N'].iloc[-1] / test_prices.iloc[0]).multiply(RP_portfolio)
    RP_values = pd.DataFrame((test_prices * RP_shares).sum(axis=1), columns=["N"])
    RP_holdings = RP_holdings.append(pd.DataFrame(np.tile(RP_shares,
                                                        (len(test_prices.index), 1)),
                                                    index=test_prices.index,
                                                    columns=tickers))

    # all the values of the portfolio
    portfolio_values = portfolio_values.append(pd.concat([MCVAR_values, SHARPE_values, MVO_values, N_values, RP_values], axis=1))

    # update to next period
    sample_end = test_end
    test_start = sample_end

    sample_start = sample_end - business_days(test_start_date - relativedelta(years=4), test_start_date)

    sample_factors = f5[sample_start:sample_end]
    sample_prices = prices[sample_start:sample_end]
    sample_volumes = volumes[sample_start:sample_end]

portfolio_values.to_csv(os.getcwd() + r'/data/VALIDATE/PORTFOLIO_RETURNS_{}.csv'.format(rs))
MCVAR_holdings.to_csv(os.getcwd() + r'/data/VALIDATE/MCVAR_HOLDINGS_{}.csv'.format(rs))
SHARPE_holdings.to_csv(os.getcwd() + r'/data/VALIDATE/SHARPE_HOLDINGS_{}.csv'.format(rs))
MVO_holdings.to_csv(os.getcwd() + r'/data/VALIDATE/MVO_HOLDINGS_{}.csv'.format(rs))
N_holdings.to_csv(os.getcwd() + r'/data/VALIDATE/N_HOLDINGS_{}.csv'.format(rs))
RP_holdings.to_csv(os.getcwd() + r'/data/VALIDATE/RP_HOLDINGS_{}.csv'.format(rs))


# we dont need ot run it again
# rs = True
# portfolio_values = pd.read_csv(os.getcwd() + r'/data/VALIDATE/{}/PORTFOLIO_RETURNS_{}.csv'.format('rs' if rs else 'no_rs', rs), index_col=0)
# MCVAR_holdings = pd.read_csv(os.getcwd() + r'/data/VALIDATE/{}/MCVAR_HOLDINGS_{}.csv'.format('rs' if rs else 'no_rs', rs), index_col=0)
# SHARPE_holdings = pd.read_csv(os.getcwd() + r'/data/VALIDATE/{}/SHARPE_HOLDINGS_{}.csv'.format('rs' if rs else 'no_rs', rs), index_col=0)
# MVO_holdings = pd.read_csv(os.getcwd() + r'/data/VALIDATE/{}/MVO_HOLDINGS_{}.csv'.format('rs' if rs else 'no_rs', rs), index_col=0)
# N_holdings = pd.read_csv(os.getcwd() + r'/data/VALIDATE/{}/N_HOLDINGS_{}.csv'.format('rs' if rs else 'no_rs', rs), index_col=0)
