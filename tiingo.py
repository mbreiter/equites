import os
import time
import requests

import pandas as pd

TIINGO_KEY = '6d2d79e31c7c1b6bae9be7e8986b4a5fe3ce5111'
TIINGO_EOD = 'https://api.tiingo.com/tiingo/daily/%s/prices?startDate=%s&endDate=%s'

def get_data(tickers, data_point, start_date, end_date, save=True, fail_safe=True):
    data = pd.DataFrame()
    start = time.time()

    try:
        for ticker in tickers:
            data[ticker] = tiingo(ticker, start_date, end_date)[data_point]

        print("\n\nSUCCESS: retrieved new %s data ..." % data_point)
        print('finished retrieving %s data in %f seconds.\n\n' % (data_point, time.time() - start))

    except:
        if fail_safe:
            data = pd.read_csv(os.getcwd() + r'/data/%s.csv' % data_point)

            print("\n\nERROR: could not retrieve new %s data ... retrieved old data" % data_point)
            print('finished retrieving %s data in %f seconds.\n' % (data_point, time.time() - start))

            return data
        else:
            print("\n\nERROR: could not retrieve new %s data." % data_point)

            return None


    data.index = pd.to_datetime(data.index)

    if save:
        data.to_csv(os.getcwd() + r'/data/%s.csv' % data_point)

    return data.dropna()


def tiingo(ticker, start_date, end_date):
    headers = {'Content-Type': 'application/json',
               'Authorization': 'Token %s' % TIINGO_KEY}

    response = requests.get(TIINGO_EOD % (ticker, start_date, end_date),
                            headers=headers).json()
    data = pd.DataFrame(response)

    return data.set_index('date')
