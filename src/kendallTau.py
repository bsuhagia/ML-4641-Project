import sys
import pandas as pd
import scipy as sp
import scipy.stats
import pylab as pl
import numpy as np
from pprint import pprint

print( sys.argv[1:] )
#a = pd.read_csv('data_for_graph/GS.csv')
#b = pd.read_csv('data_for_graph/AVB.csv')
#
#print( a )
#print( b )
#
#a_list = list(a['percent_from_mean'])
#b_list = list(b['percent_from_mean'])
#
#print( a_list )
#print( b_list )
#
#tau, p = scipy.stats.kendalltau(a_list, b_list)
#
#print tau
#print p

companies = pd.read_csv( 'constituents_with_deletions.csv' )
print( companies )

print( len( companies.Symbol ) )
symbols = companies.Symbol

symbols_averages = {}
bad_symbols = []
data_size = len( companies.Symbol )
data = [[2]*data_size for _ in range(data_size)]

for index, symbol in enumerate( symbols ):
    print( index + ' : ' + symbol )
    for c, other_symbol in enumerate( symbols ):
        if(data[index][c] == 2):
            try:
                if( symbol not in symbols_averages ):
                    print( 'READING CSV FOR ' + symbol )
                    symbol_csv = pd.read_csv('data_for_graph/'+ symbol + '.csv')
                    symbol_list = list(symbol_csv['percent_from_mean'])
                    symbols_averages[symbol] = symbol_list
                if( other_symbol not in symbols_averages ):
                    print( 'READING CSV FOR ' + other_symbol )
                    symbol_csv = pd.read_csv('data_for_graph/'+ other_symbol + '.csv')
                    symbol_list = list(symbol_csv['percent_from_mean'])
                    symbols_averages[other_symbol] = symbol_list
                tau, p = scipy.stats.kendalltau( symbols_averages[symbol], symbols_averages[other_symbol] )
                data[index][c] = tau
                data[c][index] = tau
            except:
                bad_symbols.append( other_symbol )
data_np = np.array(data)
pl.pcolor(data_np)
pl.colorbar()
pl.show()
