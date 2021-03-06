# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:50:49 2019

@author: bill-
"""

import pandas as pd
import numpy as np
import os
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
#if tokenizing error arises; might be due to pandas generated columns names with an \r
#then the discrepancy causes an error; specify separator explicitly to fix
df1 = pd.read_excel(vendor_list_input, header = 0, names = ['MerchantName',\
                                                          'BillCategory'])
print("loading the vendor list...")
BillVendors_uniqueVals = df1['MerchantName'].unique()
BillVendors = BillVendors_uniqueVals.tolist()

#print(BillVendors)

#%%
#statements = list of bank statement strings
for i in range(len(df.index)):
    descriptions = str(df.iloc[i]['shopname'])
    descriptions = descriptions.lower()
    #print(descriptions)
    for BillVendor in BillVendors:
        BillVendor = BillVendor.lower()
        #print(BillVendor)
        #condition 1 inactivated; just use string length
        cond_1 = BillVendor in descriptions
        cond_2 = len(str(BillVendor)) == len(str(descriptions))
        if np.logical_and(cond_1, cond_2) == True:
            print("Bill detected:{}".format(BillVendor))

#%%
# figure out repetitive payments
# exclude these merchants as repetitive payments
blacklist = ['Uber', 'Lyft', 'Paypal', 'E-ZPass']