# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:01:06 2020

@author: bill-
"""

'''
Purpose of this script is to interact with a csv file/database and produce a dictionary
with unique IDs and corresponding income and expenses in separate dictionaries
-determine income and expenses based on given categories
-add it either to the INCOME DICTIONARY or the EXPENSE DICTIONARY
-find out daily, weekly and monthly throughput of accounts and their excess cash
-develop a logic or daily limits and spending patterns
'''
#load needed packages
import pandas as pd
import numpy as np
#import regex
import os
#%%
#determine the path of the files
path_win = os.path.relpath(r'C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx')
#path_mac =
#read the original XLSX file and then split it up in 3 different dataframes
#no preprocessing here or encoding
df_card = pd.read_excel(path_win, sheet_name = "Card Panel")
df_bank = pd.read_excel(path_win, sheet_name = "Bank Panel")
df_demo = pd.read_excel(path_win, sheet_name = "User Demographics")
#in the non-encoded verion all columns still have correct types
#extract unique numbers from all panels to find out unique users;
card_members = df_card['unique_mem_id'].unique()
bank_members = df_bank['unique_mem_id'].unique()
demo_members = df_card['unique_mem_id'].unique()
#append these unique to dictionaries measuring expenses or income with their respective categories
#income_category = list(df_card[].unique())
#expense_category = list()
#%%
#append to lists whether it is income or expense
#Create an empty dictionary: names
income_dict = {}

#Loop over the girl names
for card_members, income in df_card.items():
    #Add each name to the names dictionary using rank as the key
    #Dict[Key] = Value
    income_dict[card_members] = income
#Sort the names list by rank in descending order and slice the first 10 items
for rank in sorted(card_members, reverse = False)[:10]:
    #Print each item
    print(income_dict[card_members])
#%%
#Import the python CSV module
import csv

#Create a python file object in read mode for the baby_names.csv file: csvfile
csvfile = open('baby_names.csv', 'r')

#Loop over a csv reader on the file object
for row in csv.reader(csvfile):
    #Print each row
    print(row)
    #Add the rank and name to the dictionary
    baby_names[row[5]] = row[3]
#Print the dictionary keys
print(baby_names.keys())
