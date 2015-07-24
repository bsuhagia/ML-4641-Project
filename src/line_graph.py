import sys
import pandas as pd
import scipy as sp
import scipy.stats
import pylab as pl
import numpy as np
from pprint import pprint

def isNan( val ):
    return val != val

constituents = pd.read_csv( 'constituents_with_deletions.csv' )
clusters = pd.read_csv( 'kmeansTickerClusters5.csv' )

pprint( clusters )

bad_symbols = []

for column in clusters:
    pprint( column )
    import pylab as p
    for symbol in clusters[column]:
        if( not isNan(symbol) ):
            try:
                print( symbol )
                symbol_csv = pd.read_csv('data_for_graph/'+ symbol + '.csv')
                symbol_list = symbol_csv['percent_from_mean']
                p.plot(symbol_list)
            except:
                bad_symbols.append( symbol )
    p.show()
pprint( clusters )
