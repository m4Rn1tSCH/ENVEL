#load JSON package
import json5

#GET /stock/{symbol}/batch
GET https://api.iextrading.com/1.0/stock/market/batch?symbols=aapl,fb,tsla&types=quote,news,chart&range=1m&last=5

# String of path to file: stocks_data_path
stocks_data_path = 'https://api.iextrading.com/1.0/stock/market/batch?symbols=aapl,fb,tsla&types=quote,news,chart&range=1m&last=5' 

#empty list to store stock prices
stocks_data = []

# Open connection to file
stocks_file = open(stocks_data_path, "r")

# read JSON output on IEX and store in list: stocks_data
##for line in stocks_file:
    ##price_data = json5.loads(line)
    ##stocks_data.append(price_data)

# Close connection to file
stocks_file.close()

# Print the keys of the first tweet dict
print(stocks_data[0].keys())



##GET /stock/{symbol}/chart/{range}
#GET /stock/{symbol}/batch
##/stock/market/batch?symbols=aapl,fb,tsla&types=quote,news,chart&range=1m&last=5
##/stock/aapl/batch?types=quote,news,chart&range=1m&last=1
##Use the /ref-data/symbols endpoint to find the symbols that we support
#https://api.iextrading.com/1.0 

#Filtered requests by sector/industry
#    /stock/market/collection/sector?collectionName=Health%20Care
#    /stock/market/collection/tag?collectionName=Computer%20Hardware
#    /stock/market/collection/list?collectionName=iexvolume


