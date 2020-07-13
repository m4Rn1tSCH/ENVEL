# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:33:46 2019

@author: bill-
"""

##input_1: csv with transactions
##input_2: xlsx with transactions
##output: list with detected bills

####PACKAGES OF THE MODULE######
import pandas as pd
import os
import re

def bill_recognition(transaction_input = relative_t_path, vendor_list = relative_v_path, exclude = blacklist):


    relative_t_path = './TransactionsD_test.csv'
    relative_v_path = './BillVendors_Only.xlsx'
    #exclude these merchants as repetetive payments
    blacklist = ['Uber', 'Lyft', 'Paypal', 'E-ZPass']
    bills_found = []

    # load files
    df = pd.read_csv(transaction_input, header = 0, names = ['date',
                                                             'category',
                                                             'trans_cat',
                                                             'subcat',
                                                             'shopname',
                                                             'amount'])

    #if tokenizing error arises; might be due to pandas generated columns names with an \r
    #then the discrepancy causes an error; specify separator explicitly to fix
    df1 = pd.read_excel(vendor_list, header = 0, names = ['MerchantName',\
                                                              'BillCategory'])

    BillVendors_uniqueVals = df1['MerchantName'].unique()
    BillVendors = BillVendors_uniqueVals.tolist()

    bills_found = []
    #statements = list of bank statement strings
    for i in range(len(df.index)):
        descriptions = str(df.iloc[i]['shopname']).lower()
        for BillVendor in BillVendors:
            #re.I makes the process ignore lower/upper case
            if re.search(BillVendor, descriptions, flags = re.I):
                # append to bill_found list
                bills_found.append(descriptions)
                print("bill found")
        else:
            print("no known bill found :(")

#iterate through the elements of bills_found
list_of_int = []
#convert the bills_found list to a tuple that is hashable
#bills_found is a key of a dictionary and is hashable
bill_dict = {tuple(list_of_int): bills_found}
for i in range(len(bills_found)):
    if re.search(exclude, bills_found, flags = re.I):
        # remove from bill_found list
        bills_found.remove(descriptions)
        print("blacklisted bill removed")
    else:
        pass
#recurring bills have breen written to a list
