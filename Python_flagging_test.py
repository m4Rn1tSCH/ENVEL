#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:47:43 2020

@author: bill
"""
# flaggin test

merch_df = pd.DataFrame(data = {'merchant_name':un_merchants})
frequency_strings = ['recurring', 'bill', 'payment', 'pmnt', 'amazon prime',
                     'amz prime', 'netflix', 'hulu', 'republic wireless',
                     'national grid', 'eversource', 'comcast', 'at&t']
pet_strings = ['petco', 'petsmart']
grocery_strings = ['wholefoods', 'wholefds', 'aldi', 'wegmann', 'food lion', 'costco',
             'target', 'market basket', 'wal mart', 'jason food mart', 'walgreens',
             'cvs']

# functions
merch_final_df = add_recurrence_col(df=merch_df, strings=frequency_strings)
merch_final_df = add_pet_col(df=merch_final_df, strings=pet_strings)
merch_final_df = add_groceries_col(df=merch_final_df, strings=grocery_strings)
#%%
# empty support column
strings = ['recurring', 'bill', 'payment', 'pmnt', 'amazon prime', 'netflix', 'hulu']
help_col = []
for index, merchant in enumerate(merch_df['merchant_name']):
    # print(index, merchant)
    for element in strings:
        # tuple has to be a single string for regex
        if re.search(element, str(merchant)):
            # print("match")
            help_col.append("rec")
        else:
            # print("no")
            help_col.append("non_rec")
help_col = pd.DataFrame(help_col,columns = ["Helper_col"])
merch_df['recurrence']= help_col['Helper_col']
#%%
# v2 empty support column
strings = ['recurring', 'bill', 'payment', 'pmnt', 'amazon prime', 'netflix', 'hulu']
merch_df['recurrence'] = ""
for index, merchant in enumerate(merch_df['merchant_name']):
    # print(index, merchant)
    for element in strings:
        # tuple has to be a single string for regex
        if re.search(element, str(merchant)):
            merch_df.loc[index,'recurrence'] = "1"
        else:
            merch_df.loc[index,'recurrence'] = "0"
