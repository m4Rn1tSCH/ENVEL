import pandas as pd
import FinanceDataReader as fdr 

tickers_NYSE = fdr.StockListing("NYSE")
tickers_NASDAQ = fdr.StockListing("NASDAQ")
tickers_SP500 = fdr.StockListing("SP500")
tickers_KRX = fdr.StockListing("KRX")
tickers_KOSDAQ = fdr.StockListing("KOSDAQ")

df_NYSE = pd.DataFrame(tickers_NYSE)
df_NASDAQ =pd.DataFrame(tickers_NASDAQ)
df_SP500 = pd.DataFrame(tickers_SP500)
df_KRX = pd.DataFrame(tickers_KRX)
df_KOSDAQ = pd.DataFrame(tickers_KOSDAQ)

# Create a Pandas Excel writer
#THIS SETS THE NAME OF THE EXCEL FILE
writer = pd.ExcelWriter('FINREA_NYSE_NASDAQ_SP500.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet
df_NYSE.to_excel(writer, sheet_name='NYSE')
df_NASDAQ.to_excel(writer, sheet_name='NASDAQ')
df_SP500.to_excel(writer, sheet_name='S&P_500')
df_KRX.to_excel(writer, sheet_name='KRX')
df_KOSDAQ.to_excel(writer, sheet_name='KOSDAQ')

# Close the Pandas Excel writer and save the Excel file
#WILL BE SAVED IN THE SRIPT'S FILE LOCATION
writer.save()
