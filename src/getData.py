import pandas as pd
import pandas.io.data as web
from pandas.io.data import DataReader
from datetime import datetime

start = datetime(2010, 6, 10)
end = datetime(2015, 6, 10)

constituents =  pd.read_csv( "constituents.csv" )

symbols =  constituents['Symbol']
names =  constituents['Name']

for index, symbol in enumerate( symbols ):
    name = names[index]

    name = name.replace("-", "")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace("&", "")
    name = name.replace("'", "")
    name = name.replace(" ", "")
    try:
        print symbol
        print name
        print()
        data = web.DataReader(symbol, 'yahoo', start, end)
        percent_diff = []
        a = list(data.columns.values)
        averages = data.mean()
        adj_average = averages['Adj Close']
        data = ( data.assign(percent_from_mean = ( data['Adj Close'] - adj_average ) / adj_average * 100 ))
        data.to_csv( "data_for_graph/" + symbol + ".csv" )
    except IOError:
        print( "FAILED PARSING " + symbol )
        pass
