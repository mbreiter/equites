from __future__ import print_function
import torch
import pandas as pd 
import numpy as np 
import server.models.portfolio.util as ut
import ast
from sklearn.preprocessing import MinMaxScaler

class TextClassDataLoader(object):

	def __init__(self, path_file, word_to_index = None, batch_size = 32, predict = False, check_ml = None):

		self.batch_size = batch_size
		#self.word_to_index = word_to_index
		self.scaler = MinMaxScaler()
		
		df = pd.read_csv('server/models/portfolio/data/test.csv')
		df = df[['prices', 'return2']]
		df.prices = df.prices.apply(lambda x: ast.literal_eval(x))
		#df['body'] = df['body'].apply(ut._tokenize)
		#df['body'] = df['body'].apply(self.generate_indexifyer())

		self.samples = df.values.tolist()
		all_values = []
		for x in range(len(self.samples)):
			temp = self.samples[x][0] + [self.samples[x][1]]
			all_values += temp
		all_values = np.asarray(all_values).reshape(-1, 1)
		self.scaler.fit(all_values)
		prices = self.scaler.transform(df.prices.values.tolist())
		return2 = self.scaler.transform(np.asarray(df.return2.values).reshape(-1,1))
		for i in range(len(self.samples)):
			self.samples[i][0] = prices[i]
			self.samples[i][1] = return2[i][0]

		self.n_samples = len(self.samples)
		self.n_batches = int(self.n_samples / self.batch_size)
		#self.max_length = self._get_max_length()
		self._shuffle_indices()

		self.report()
		if predict:
			self.predict_batches = (self.samples, check_ml.mul(pd.DataFrame(1 + np.random.uniform(-0.05, 0.1, len(tickers)), index=check_mu.index, columns=check_mu.columns))) 

	def _shuffle_indices(self):
		self.indices = np.random.permutation(self.n_samples)
		self.index = 0
		self.batch_index = 0

	def generate_indexifer(self):

		def indexify(lst_text):
			indices = []
			for word in lst_text:
				if word in self.word_to_index:
					indices.append(self.word_to_index[word])
				else:
					indices.append(self.word_to_index['__UNK__'])
			return indices
		return indexify

	@staticmethod
	def _padding(batch_x):
		batch_s = sorted(batch_x, key = lambda x: len(x))
		size = len(batch_s[-1])
		for i, x in enumerate(batch_x):
			missing = size - len(x)
			batch_x[i] = batch_x[i] + [0 for _ in range(missing)]
		return batch_x

	def _create_batch(self):
		batch = []
		n = 0
		while n < self.batch_size:
			_index = self.indices[self.index]
			#self.samples[_index][0] = self.scaler.transform(self.samples[_index][0])
			#self.samples[_index][1] = self.scaler.transform(self.samples[_index][1])
			batch.append(self.samples[_index])
			self.index += 1
			n += 1
		self.batch_index += 1

		string, label = tuple(zip(*batch))
		#seq_lengths = torch.LongTensor(list(map(len, string)))
		length = len(string[0])
		seq_tensor = torch.zeros(len(string), length).long()
		for idx, seq in enumerate(string):
			seq_tensor[idx, :length] = torch.FloatTensor(seq)

		seq_tensor = seq_tensor.transpose(0, 1)
		#seq_lengths, perm_idx = seq_lengths.sort(0, descending = True)
		#seq_tensor = seq_tensor[perm_idx]
		# seq_tensor = seq_tensor.transpose(0, 1)
		label = torch.LongTensor(label)
		#label = label[perm_idx]

		return seq_tensor, label


	def __len__(self):
		return self.n_batches

	def __iter__(self):
		self._shuffle_indices()
		for i in range(self.n_batches):
			if self.batch_index == self.n_batches:
				raise StopIteration()
			yield self._create_batch()

	def show_samples(self, n = 10):
		for sample in self.samples[:n]:
			print(sample)

	def report(self):
		print('# samples: {}'.format(len(self.samples)))
		#print('max len: {}'.format(self.max_length))
		#print('# vocab: {}'.format(len(self.word_to_index)))
		print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))
