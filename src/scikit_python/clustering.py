import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import finance
from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold


import pandas as pd

d1 = datetime.datetime(2010, 06, 10)
d2 = datetime.datetime(2015, 06, 10)

stock_list = pd.read_csv('updatedconstituents.csv')
stock_list = stock_list.drop('Sector', 1)
# print stock_list



symbols_all, names_all = stock_list.Symbol.values, stock_list.Name.values


print symbols_all

print(type(symbols_all))
# print 'this are the symbols'
# print symbols

# print 'this are the names'
# print names
ticker_index = 1
for symbol in symbols_all:
    # print 'starting...', ticker_index
    quotes = [finance.quotes_historical_yahoo_ochl(symbol, d1, d2, asobject=True)]
    # print 'working...', ticker_index
    # ticker_index = ticker_index + 1

# quotes = [finance.quotes_historical_yahoo_ochl(symbol, d1, d2, asobject = True) for symbol in symbols_all]

# print quotes

open = np.array([q.open for q in quotes]).astype(np.float)
# print open

close = np.array([q.close for q in quotes]).astype(np.float)
# print close

variation = close - open
# print variation




edge_model = covariance.GraphLassoCV()
X = variation.copy().T
X /= X.std(axis = 0)

# uncomment and check if this works
# this is similar to the example on scikit

edge_model.fit(X)

# _ , labels = cluster.affinity_propagation(edge_model.covariance_)
# n_labels = labels.max()

# for i in range(n_labels + 1):
#     print('Cluster %i: %s' % ((i+1), ','.join(names[labels == i])))

