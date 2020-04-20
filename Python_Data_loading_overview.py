#INPUT: CSV-FILE OR XLSX-FILE
#OUTPUT: PANDAS DATA FRAME


#Packages
import os
import pandas as pd
import Python_random_sample as rs

#switching the backslashes to slashes
#letter strings can remmain
link = r"C:\Users\bill-\Desktop\TransactionsD.csv"
new_link = link.replace(os.sep, '/')
file = ''.join(('', new_link,''))

#load the data and skip the first row, then rename the columns to something informative
#columns
#date = date of transaction
#trans_cat = category of transaction
#subcat = subcategory
#shopname = shop name
#amount = amount in USD
data = pd.read_csv(file, skiprows = 1, index_col = None, names = ['category', 'trans_cat', 'subcat', 'shopname', 'amount'])

#draw random sample

transaction_sample = rs.draw_sample(data, sample_size = 32, sample_weights = None, random_state = None)
