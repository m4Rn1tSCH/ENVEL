import pandas as pd
import quandl as quandl
import xlsxwriter as xlsx
## configuration of the unique API key
## NEED TO BE DONE FOR EACH NEW USER!
## path= 'C:/Users/bill-/OneDrive/Dokumente/Docs Bill/TA_files/functions_scripts'
quandl.ApiConfig.api_key = "LUDzkxDAmGH31z-p-csu"
print("API-key entered, data is being retrieved...")

LME_data = quandl.get('LME/PR_CO', start_date='2013-01-01', end_date='2018-11-12')
GDP_data = quandl.get('FRED/GDP', api_key ='LUDzkxDAmGH31z-p-csu')

# Create the Pandas dataframes from stock data
df_LME = pd.DataFrame(LME_data)
df_GDP = pd.DataFrame(GDP_data)

print(df_LME.head())
print(df_GDP.head())

# Create a Pandas Excel writer
#THIS SETS THE NAME OF THE EXCEL FILE
writer = pd.ExcelWriter('QUANDL_GDP_LME.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet
df_LME.to_excel(writer, sheet_name='LME_Co_prices')
df_GDP.to_excel(writer, sheet_name='US_GDP')

# Close the Pandas Excel writer and save the Excel file
#WILL BE SAVED IN THE SRIPT'S FILE LOCATION
writer.save()
