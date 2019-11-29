import os
import re
import time
import requests

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup

YAHOO = 'https://finance.yahoo.com/quote/%s?p=%s'


def get_mkt_cap(tickers, save):
    data = {}

    try:
        for asset in tickers:
            data[asset] = mkt_cap(asset)

        print("\n\nSUCCESS: retrieved new market cap values ...")
    except:
        data = pd.read_csv(os.getcwd() + r'/data/mkt_cap.csv', index_col=0)

        print("\n\nERROR: failed to get new market cap values ... using old values")

        return data

    mktcap = pd.DataFrame(data.values(), index=data.keys(), columns=['MKT'])

    if save:
        mktcap.to_csv(os.getcwd() + r'/data/mkt_cap.csv')

    return mktcap

def mkt_cap(ticker):
    url = YAHOO % (ticker, ticker)

    html = str(BeautifulSoup(requests.get(url).content, 'html5lib'))
    cap = re.search('"totalAssets":{"raw":(.*?),"fmt":"', html)

    return float(cap.group(1))


def get_mkt_weights(mkt_cap):
    return mkt_cap / mkt_cap.sum()


def bl(tickers, l, tau, mktcap, Sigma, P, Omega, q, adjust):

    start = time.time()

    if adjust:
        mktcap = mktcap.mul(1 + q)

    mktwgt = get_mkt_weights(mktcap)
    pi = l * Sigma.dot(mktwgt.values)

    Sigma_bl = pd.DataFrame(np.linalg.inv(np.linalg.inv(tau * Sigma) + P.T.dot(np.linalg.inv(Omega)).dot(P)),
                            index = tickers,
                            columns = tickers)
    mu_bl = Sigma_bl.dot(np.linalg.inv(tau * Sigma).dot(pi) + P.T.dot(np.linalg.inv(Omega)).dot(q))
    mu_bl.columns = ["returns"]

    print('finished calculating bl parameters in %f seconds.\n\n' % (time.time() - start))

    return mu_bl, Sigma_bl
