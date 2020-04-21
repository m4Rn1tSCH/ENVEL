import schedule
import time
import pandas as pd
import pandas_datareader as pdr
import Python_JSON_reader_IEXtrading as iex
import Python_IEX_xlsx_exp as xlsx


##NOT WORKING
def export_data(message='IEX data 1m intervall is being retrieved...'):
	xlsx.export_df()

schedule.every(5).seconds.do(export_data)

while True:
	schedule.run_pending()
	time.sleep(0.001)

##WORKING
def pull_data(message='IEX data 1m intervall is being retrieved...'):
	print(message)
	iex.pull_data_1m()

schedule.every(5).minutes.do(pull_data)

while True:
	schedule.run_pending()
	time.sleep(0.001)


