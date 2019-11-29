import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd 
import numpy as np 
from datetime import datetime
from scipy.stats.mstats import gmean
from dateutil.relativedelta import relativedelta
import os

from stats import *
from cost import costs
from tiingo import get_data
from optimize import optimize
from bl import bl, get_mkt_cap
from rs import fama_french, regime_switch, current_regime, business_days, expected_returns, covariance



class RNN(nn.Module):

	def __init__(self, embed_size,
				 num_output,
				 rnn_model = 'GRU',
				 use_last = True,
				 padding_index = 0,
				 hidden_size = 64,
				 num_layers = 1,
				 batch_first = True):

		super(RNN, self).__init__()
		self.use_last = use_last
		self.drop_en = nn.Dropout(p = 0.6)

		self.end_date = datetime.now().strftime("%Y-%m-%d")
		self.start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - relativedelta(years=6)).strftime("%Y-%m-%d")
		self.tickers = list(pd.read_csv(os.getcwd() + r'/data/tickers.csv')['Tickers'])
		
		if rnn_model == 'LSTM':
			self.rnn = nn.LSTM(input_size = embed_size, hidden_size = hidden_size,
							   num_layers = num_layers, dropout = 0.5,
							   batch_first = True, bidirectional = False)
		elif rnn_model == 'GRU':
			self.rnn = nn.GRU(input_size = embed_size, hidden_size = hidden_size,
							  num_layers = num_layers, dropout = 0.5,
							  batch_first = True, bidirectional = False)

		self.bn2 = nn.BatchNorm1d(int(hidden_size))
		self.fc = nn.Linear(int(hidden_size), int(num_output))


	def forward(self, x):
		#x_embed = self.drop_en(x)
		#x_embed = nn.functional.dropout(x)
		x_embed = x.view(18, x.shape[1], -1)
		#packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(), batch_first = True)
		x_embed = x_embed.type(torch.FloatTensor)
		packed_output, ht = self.rnn(x_embed, None)
		#out_rnn, _ = pad_packed_sequence(packed_output, batch_first = True)

		#row_indices = torch.arange(0, x.size(0)).long()
		#col_indices = seq_lengths - 1
		#if next(self.parameters()).is_cuda():
		#	row_indices = row_indices.cuda()
			#col_indices = col_indices.cuda()
		#if self.use_last:
			#last_tensor = out_rnn[row_indices, col_indices, :]
			#last_tensor = packed_output[row_indices, :]
		#else:
			#last_tensor = out_rnn[row_indices, :, :]
			#last_tensor = packed_output[row_indices, :]
			#last_tensor = torch.mean(last_tensor, dim = 1)
#change labels to predict returns from stock price, but output mu_ml (do this in run_optimization - move it outside)
		fc_input = self.bn2(packed_output[-1].view(x.shape[1], -1))
		out = self.fc(fc_input)
		#out = self.run_optimization(self.end_date, self.start_date, out)
		return out.view(-1)


	def run_optimization(self, end_date, start_date, mu_ml):
		rebalance_date = (datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(months=6, days=1)).strftime("%Y-%m-%d")
		rebalance_date = datetime.strftime(pd.bdate_range(end_date, rebalance_date)[-1], "%Y-%m-%d")
		prices = get_data(self.tickers, 'adjClose', start_date, end_date, save=False)
		factors = fama_french(start_date, end_date, save=False)
		returns = (prices / prices.shift(1) - 1).dropna()[:len(factors)]
		R = returns.values
		
		## *********************************************************************************************************************
		#  factor model
		## *********************************************************************************************************************

		factors.drop('RF', axis=1, inplace=True)
		F = np.hstack((np.atleast_2d(np.ones(factors.shape[0])).T, factors))
		transmat, loadings, covarainces = regime_switch(R, F, self.tickers)
		baseline = 30
		regime = current_regime(R, F, loadings, baseline)


		# get the number of days until the next scheduled rebalance
		days = business_days((datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(days=1)).strftime("%Y-%m-%d"), rebalance_date)

		# get the estimate returns and covariances from the factor model
		mu_rsfm = pd.DataFrame(days * expected_returns(F, transmat, loadings, regime), index=self.tickers, columns=['returns'])
		cov_rsfm = pd.DataFrame(days * covariance(R, F, transmat, loadings, covarainces, regime), index=self.tickers, columns=self.tickers)

		# write estimates to a csv file
		mu_rsfm.to_csv(os.getcwd() + r'/data/mu_rsfm.csv')
		cov_rsfm.to_csv(os.getcwd() + r'/data/cov_rsfm.csv')
		mktcap = get_mkt_cap(self.tickers, save=True)

		# calculate the market coefficient
		l = (gmean(factors.iloc[-days:,:]['MKT'] + 1,axis=0) - 1)/factors.iloc[-days:,:]['MKT'].var()

		mu_bl1, cov_bl1 = bl(tickers=self.tickers,
		                     l=l, tau=1,
		                     mktcap=mktcap,
		                     Sigma=returns.iloc[-days:,:].cov().values * days,
		                     P=np.identity(len(self.tickers)),
		                     Omega=np.diag(np.diag(cov_rsfm)),
		                     q=mu_rsfm.values,
		                     adjust=False)

		#mu_ml = mu_bl1.mul(pd.DataFrame(1 + np.random.uniform(-0.05, 0.1, len(tickers)), index=mu_bl1.index, columns=mu_bl1.columns))
		mu_ml = pd.DataFrame(mu_ml, columns = ['returns'])
		mu_bl2, cov_bl2 = bl(tickers=self.tickers,
		                     l=l, tau=1,
		                     mktcap=mktcap,
		                     Sigma=returns.iloc[-days:,:].cov().values * days,
		                     P=np.identity(len(self.tickers)),
		                     Omega=np.diag(np.diag(cov_rsfm)),
		                     q=mu_ml.values,
		                     adjust=True)

		cost = costs(tickers=self.tickers,
		             cov=cov_rsfm,
		             prices=prices.iloc[-2, :] if prices.iloc[-1, :].isnull().values.any() else prices.iloc[-1, :],
		             start_date=(datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(years=1)).strftime("%Y-%m-%d"),
		             end_date=end_date,
		             alpha=5)

		risk_tolerance = [((1, 10), (0, 0.10)),
	                  ((5, 5), (0, 0.20)),
	                  ((10, 1), (-0.05, 0.30))]

		soln = optimize(mu = (mu_bl1.values.ravel(), mu_bl2.values.ravel()),
		                sigma = (cov_bl1.values, cov_bl2.values),
		                alpha = (0.05, 0.10),
		                return_target = (0.05, 0.05),
		                costs = cost,
		                prices = prices.iloc[-2, :].values if prices.iloc[-1, :].isnull().values.any() else prices.iloc[-1, :].values,
		                gamma = risk_tolerance[2])

		x1 = pd.DataFrame(soln.x[:int(len(mu_bl1))], index=mu_bl1.index, columns=['weight'])
		x2 = pd.DataFrame(soln.x[int(len(mu_bl2)):], index=mu_bl2.index, columns=['weight'])
		print('\n\n********************************************************************')
		print('\tperiod one results')
		print('********************************************************************\n')

		#print(x1)

		print("\nportfolio return: %f" % (ret(mu_bl1, x1) * 100))
		print("portfolio volatility: %f" % (vol(cov_bl1, x1) * 100))
		print("portfolio var%f: %f" % (1-0.05, var(mu_bl1, cov_bl1, 0.05, x1)))
		print("portfolio cvar%f: %f" % (1-0.05, cvar(mu_bl1, cov_bl1, 0.05, x1)))


		print('\n\n********************************************************************')
		print('\tperiod two results')
		print('********************************************************************\n')

		#print(x2)

		print("\nportfolio return: %f" % (ret(mu_bl2, x1) * 100))
		print("portfolio volatility: %f" % (vol(cov_bl2, x1) * 100))
		print("portfolio var%f: %f" % (1-0.05, var(mu_bl2, cov_bl2, 0.05, x1)))
		print("portfolio cvar%f: %f" % (1-0.05, cvar(mu_bl2, cov_bl2, 0.05, x1)))

		return (ret(mu_bl2, x1) * 100)


