from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import pandas as pd

ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')

data.to_csv('stock_data.csv')

