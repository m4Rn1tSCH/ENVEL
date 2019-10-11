
import pandas as pd

#INPUT: PANDAS DATA FRAME
#OUTPUT: SUBSET OF A PANDAS DATA FRAME

#pick a column that contains the words Shop name or transaction or similar combinations


columns = file_in.columns
column_list = ['SHOPNAME', 'shopname', 'ShopName', 'transactions', 'Transactions', 'History', 'Transaction History']
for x in column_list:
    if columns.str.contains(x).any():
        df_transactions = pd.DataFrame(file_in[x])
        
print(df_transactions)
