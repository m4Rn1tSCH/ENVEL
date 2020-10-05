#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:26:55 2020

@author: bill
"""


import pandas as pd 
import numpy as np
import os 
import regex as re
from datetime import datetime as dt

def load_Vendors(filename):
	path = os.getcwd()
	csv_path = os.path.abspath(os.path.join(path + filename + ".csv"))
	Vendor_df = pd.read_csv(csv_path)
	print(f'Successfully imported {filename}.csv into pandas df')
	return Vendor_df

def save_data(filename,data):
	raw = os.getcwd()
	date_of_creation = dt.today().strftime('%d%m%Y%H%M')
	csv_path = os.path.abspath(os.path.join(raw + date_of_creation + '_' + filename + '.csv'))
	data.to_csv(csv_path)
	print(f"{filename}.csv saved to {csv_path}")


def Clean_Text(Vendor_list, Cleaning_column = 'merchant_name'):

	CC = Cleaning_column

	states = [
	'CA','TX','NY','FL','PA','OH','IL','GA','WA','NC',
	'MD','VA','NJ','MI','IN','KY','AZ','CO','TN','SC',
	'MO','AL','NV','KS','DC','WI','OK','OR','DE','LA',
	'CT','AR','MN','UT','NM','HI','MS','WV','AK','NE',
	'ID','RI','NH','IA','ME','MT','SD','ND','WY','VT',
	'PR','MP','GU','VI','NAN','MA'
	]

	cities = load_Vendors('180920201415_web_cities')['City_Name'].str.lower().values.tolist()

	#remove all numbers and special characters from the column
	Vendor_list['Merchant_noNUM'] = Vendor_list[CC].str.replace('\d+|-|!|#|$|%|^|\*|;|:|,|\.|~|-|=|_|/', " ")

	#make all char lower case
	Vendor_list['Merchant_LCase'] = Vendor_list['Merchant_noNUM'].str.lower()

	#remove words consisting of repeating X
	Vendor_list['Merchant_noX'] = Vendor_list["Merchant_LCase"].str.replace(u"x{2,}", "")

	#remove shortend words
	abrv= {'th':'','st':'','rd':'','nd':''}
	Vendor_list['Merchant_noCHAR'] = Vendor_list['Merchant_noX'].str.split(' ',expand=True).replace(abrv,regex=False).fillna('').apply(' '.join,1)

	#remove non-contributing words
	abrv= {'debit':'',
	'card':'',
	'purchase':'',
	'com':'',
	'pos':'',
	'atm':'', 
	'withdrawal':'',
	'pin':'',
	'checkcard':'',
	'check':'',
	'transfer':'',
	'to':'',
	'funds':'',
	}
	Vendor_list['Merchant_noCHAR'] = Vendor_list['Merchant_noCHAR'].str.split(' ',expand=True).replace(abrv,regex=False).fillna('').apply(' '.join,1)

	#remove single char substrings
	f = lambda x: ' '.join([item for item in x.split() if len(item) > 1])
	Vendor_list["Merchant_noSingles"] = Vendor_list["Merchant_noCHAR"].apply(f)

	#remove states abbreviation
	dictOfstates = { i.lower() : "" for i in states }
	Vendor_list['Merchant_noState'] = Vendor_list['Merchant_noSingles'].str.split(' ',expand=True).replace(dictOfstates,regex=False).fillna('').apply(' '.join,1)

	#remove city names
	Vendor_list['Merchant_ContainsCity'] = Vendor_list['Merchant_noState'].str.replace(r'\b|\b'.join(cities),"",regex=True).fillna('')

	#remove extra spaces
	Vendor_list['Merchant_Final'] = Vendor_list['Merchant_ContainsCity'].str.replace('\s+', ' ', regex=True).str.strip()


	return Vendor_list

print(dt.today().strftime('%d/%m/%Y - %H:%M'))

display_rows = 200
pd.set_option('display.max_rows', display_rows)

###########################################
### 	 Cleaning transaction data		###
###########################################

Random_trans = load_Vendors("170920201100_Random_transactions")
Random_trans = Random_trans[0:100]

Tran_data = Random_trans[['description','transaction_category_name','primary_merchant_name']]
Cleaned_Data = Clean_Text(Tran_data,'description')

# print(Cleaned_Data[['Merchant_noState','Merchant_ContainsCity','Merchant_Final']].head(display_rows))


# ###########################################
# ### 	    Unique Bill vendors     	###
# ###########################################

Bill_Vendor_list = load_Vendors("BillVendors_Only copy")
del Bill_Vendor_list['BillCategory']

Bill_Vendor_list.rename(columns = {'MerchantName':'merchant_name'}, inplace = True) 

Unique_Bill_Vendors = Clean_Text(Bill_Vendor_list)

#remove duplicates and NaN
Unique_Bill_Vendors = Unique_Bill_Vendors.drop_duplicates(subset='Merchant_Final', keep='first')
Unique_Bill_Vendors['Merchant_Final'].replace('', np.nan, inplace=True)
Unique_Bill_Vendors.dropna(subset=['Merchant_Final'], inplace=True)

# print(Unique_Bill_Vendors[['merchant_name','Merchant_Final']].head(display_rows))

# ###########################################
# ### 	       Create Bill flag           ###
# ###########################################


bill_vendors = Unique_Bill_Vendors['Merchant_Final'].values.tolist()

Cleaned_Data['Merchant_Bill_flag'] = Cleaned_Data['Merchant_Final'].str.contains(r'\b|\b'.join(bill_vendors),regex=True).astype(int)

print(Cleaned_Data[['description','Merchant_Final','Merchant_Bill_flag']].head(display_rows))