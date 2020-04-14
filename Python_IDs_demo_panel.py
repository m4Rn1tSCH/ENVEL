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
from collections import Counter
import matplotlib.pyplot as plt

transactions_win = os.path.relpath(r'C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\DEMO_PANEL_FULL.csv')
transactions_mac = os.path.relpath(r'/Users/bill/OneDrive - Envel/user_demographic.csv')
demo_full = pd.read_csv(transactions_win,
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
#%%
#Pie chart transaction type -works
demo_ct = Counter(list(demo_full['state']))
#asterisk look up, what is that?
labels, values = zip(*demo_ct.items())
#Pie chart, where the slices will be ordered and plotted counter-clockwise
#figsize (20,15) works best for legend on the right side and keeps it readable
fig1, ax1 = plt.subplots(figsize = (20, 15))
ax1.pie(values, labels = labels, autopct = '%1.1f%%',
        shadow = True, startangle = 90)
#Equal aspect ratio ensures that pie is drawn as a circle.
ax1.axis('equal')
ax1.legend(loc = 'upper right')
plt.show()