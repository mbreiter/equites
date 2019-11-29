import os

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from tiingo import get_data

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
    factors['RF'] = factors['RF'] * 100

    factors = factors.loc[start_date:end_date,:].iloc[1:]
    factors.index = pd.to_datetime(factors.index)

    return factors

## *********************************************************************************************************************
# parameters
## *********************************************************************************************************************

# get the list of assets
tickers = list(pd.read_csv(os.getcwd() + r'/data/tickers.csv')['Tickers'])[2:3]
N = len(tickers)

MAE = pd.DataFrame()
MSE = pd.DataFrame()
RMSE = pd.DataFrame()
R2 = pd.DataFrame()

for i in range(20):
    start_date = (datetime.now() - relativedelta(years=i+1)).strftime("%Y-%m-%d")
    end_date = (datetime.now() - relativedelta(years=i)).strftime("%Y-%m-%d")

    prices = get_data(tickers, 'adjClose', start_date, end_date, save=True, fail_safe=True)

    # test the fama-french 3 and 5 model, along with the CAPM model
    f5 = fama_french(start_date, end_date, five=True)
    f3 = fama_french(start_date, end_date, five=False)
    capm = f3['MKT']

    returns = (prices / prices.shift(1) - 1).dropna()[:len(f5)]
    R = returns.values - f5['RF'].values[:, None]

    f3.drop('RF', axis=1, inplace=True)
    f5.drop('RF', axis=1, inplace=True)

    # Regression
    X5 = f5.values
    X3 = f3.values
    XC = capm.values

    regressor5 = LinearRegression()
    X5_train, X5_test, R_train, R_test = train_test_split(X5, R, test_size=0.5, random_state=0)
    regressor5.fit(X5_train, R_train)
    R5_pred = regressor5.predict(X5_test)

    regressor3 = LinearRegression()
    X3_train, X3_test, R_train, R_test = train_test_split(X3, R, test_size=0.5, random_state=0)
    regressor3.fit(X3_train, R_train)
    R3_pred = regressor3.predict(X3_test)

    regressorC = LinearRegression()
    XC_train, XC_test, R_train, R_test = train_test_split(XC.reshape(-1,1), R, test_size=0.5, random_state=0)
    regressorC.fit(XC_train, R_train)
    RC_pred = regressorC.predict(XC_test)

    MAE.loc[end_date, 'CAPM'] = metrics.mean_absolute_error(R_test, RC_pred)
    MAE.loc[end_date, 'FF3'] = metrics.mean_absolute_error(R_test, R3_pred)
    MAE.loc[end_date, 'FF5'] = metrics.mean_absolute_error(R_test, R5_pred)

    MSE.loc[end_date, 'CAPM'] = metrics.mean_squared_error(R_test, RC_pred)
    MSE.loc[end_date, 'FF3'] = metrics.mean_squared_error(R_test, R3_pred)
    MSE.loc[end_date, 'FF5'] = metrics.mean_squared_error(R_test, R5_pred)

    RMSE.loc[end_date, 'CAPM'] = np.sqrt(metrics.mean_squared_error(R_test, RC_pred))
    RMSE.loc[end_date, 'FF3'] = np.sqrt(metrics.mean_squared_error(R_test, R3_pred))
    RMSE.loc[end_date, 'FF5'] = np.sqrt(metrics.mean_squared_error(R_test, R5_pred))

    R2.loc[end_date, 'CAPM'] = metrics.r2_score(R_test, RC_pred)
    R2.loc[end_date, 'FF3'] = metrics.r2_score(R_test, R3_pred)
    R2.loc[end_date, 'FF5'] = metrics.r2_score(R_test, R5_pred)

# # sample data
# sample_start = 0
# sample_days = business_days(f5.index[sample_start], f5.index[sample_start] + relativedelta(months=6))
# sample_end = sample_start + sample_days
#
# sample_5 = f5.iloc[sample_start:sample_end, ].values
# sample_3 = f3.iloc[sample_start:sample_end, ].values
#
# test_start = sample_end
# init_test_start = test_start
#
# done = False
#
# R5, R3 = [], []
#
# while test_start < len(f5) and not done:
#
#     test_days = business_days(f5.index[test_start], f5.index[test_start] + relativedelta(months=6))
#     test_end = test_start + test_days
#
#     if test_end > len(f5):
#         test_end = len(f5)
#         done = True
#
#     test_5 = f5.iloc[test_start:test_end, ].values
#     test_3 = f3.iloc[test_start:test_end, ].values
#
#     beta_5 = np.linalg.inv(sample_5.T.dot(sample_5)).dot(sample_5.T).dot(R[sample_start:sample_end, :])
#     beta_3 = np.linalg.inv(sample_3.T.dot(sample_3)).dot(sample_3.T).dot(R[sample_start:sample_end, :])
#
#     R5 += list(test_5.dot(beta_5).ravel())
#     R3 += list(test_3.dot(beta_3).ravel())
#
#     # test data
#     sample_start = test_start
#     sample_end = test_end
#
#     sample_5 = f5.iloc[sample_start:sample_end, ]
#     sample_3 = f3.iloc[sample_start:sample_end, ]
#
#     test_start = sample_end
#
# returns = pd.DataFrame(R[init_test_start:,], index=f5.iloc[init_test_start:, ].index, columns=["Realized Returns"])
# returns_5 = pd.DataFrame(R5, index=f5.iloc[init_test_start:, ].index, columns=["Returns under FF5"])
# returns_3 = pd.DataFrame(R3, index=f3.iloc[init_test_start:, ].index, columns=["Returns under FF3"])






