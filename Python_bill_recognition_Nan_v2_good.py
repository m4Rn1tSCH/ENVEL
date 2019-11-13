# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:26:23 2019

@author: bill-
"""

import pandas as pd
import os
import re
##keep this for tests on other OSs and to avoid paths problems
os.getcwd()
#%%
#Pay attention if it is a CSV or Excel file to avoid tokenization errors and separator errors
#link for the transaction csv
transaction_input = r"C:\Users\bill-\Desktop\TransactionsD.csv"
path_1 = transaction_input.replace(os.sep,'/')
transaction_input = ''.join(('', path_1, ''))
#%%
#
vendor_list_input = r"C:\Users\bill-\Dropbox\Nan\Archived\BillVendors_Only.xlsx"
path_11 = vendor_list_input.replace(os.sep,'/')
vendor_list_input = ''.join(('', path_11, ''))
#%%
#load csv
df = pd.read_csv(transaction_input, header = 0, names = ['date',
                                                         'category',
                                                         'trans_cat',
                                                         'subcat',
                                                         'shopname',
                                                         'amount'])
df.head(n = 3)
len(df.index)
#%%
# figure out repetitive payments
# exclude these merchants as repetitive payments
blacklist = ['Uber', 'Lyft', 'Paypal', 'E-ZPass']
#%%
#if tokenizing error arises; might be due to pandas generated columns names with an \r
#then the discrepancy causes an error; specify separator explicitly to fix
df1 = pd.read_excel(vendor_list_input, header = 0, names = ['MerchantName',\
                                                          'BillCategory'])
print("loading the vendor list...")
BillVendors_uniqueVals = df1['MerchantName'].unique()
BillVendors = BillVendors_uniqueVals.tolist()

#change the elements to lower case only
for BillVendor in BillVendors:
    BillVendor = BillVendor.lower()
bills_found = []
#%%
#2 conditions added here!!
#statements = list of bank statement strings
for i in range(len(df.index)):
    descriptions = str(df.iloc[i]['shopname']).lower()
    #descriptions = descriptions.lower()
    #print(descriptions)
    #print(BillVendor)
    for BillVendor in BillVendors:
        if re.search(BillVendor, descriptions):
            # append to bill_found list
            bills_found = bills_found.append(descriptions)
            print("bill found")
    else:
        print("no bills found!")
