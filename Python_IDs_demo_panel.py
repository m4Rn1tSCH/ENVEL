# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:19:59 2020

@author: bill-
"""

'''
This module pulls all unique IDs from the full demographic panel
and then passes it as a list/array
'''

import os
import pandas as pd

transactions = os.path.relpath(r'C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\DEMO_PANEL_FULL.csv')
demo_full = pd.read_csv(transactions,
                        index_col = None,
                        header = None,
                        usecols = [0, 1, 2, 4, 5, 6],
                        names = ['unique_mem_id', 'state', 'city', 'income_class', 'file_created_date', 'panel_file_created_date',])
pd.to_datetime(demo_full['file_created_date'])
pd.to_datetime(demo_full['panel_file_created_date'])
demo_full.fillna(value = 'unknown')

#first 25 users as array and list
id_array = demo_full['unique_mem_id'].unique()[:5]
id_list = list(demo_full['unique_mem_id'].unique()[:5])

#print(demo_full.head(10))