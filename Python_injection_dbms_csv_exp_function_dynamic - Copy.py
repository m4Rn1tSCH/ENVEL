# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:37:29 2020

@author: bill-
"""

'''
This module conducts the filtering of dataframes per unique user ID
Following steps are involved:
    adding feature column:"income/expense"
    summing up values per user and transaction class
    calculates injection needed before first income is received
'''
import numpy as np
import pandas as pd
import csv
from Python_IDs_demo_panel import id_list, id_array

#as a self function to use as method
def injector(self):
    #test_path = r'C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx'
    #relative path to test the file sitting directly in the folder with the script
    #test_path_2 = './2020-01-28 envel.ai Working Class Sample.xlsx'

    #df_card = pd.read_excel(os.path.abspath(test_path), sheet_name = "Card Panel")
    #card_members = df_card['unique_mem_id'].unique()
    #%%
    '''
    Filter for dataframes to find out income and expenses narrowed down to the user id
    '''
    #filter with ilocation and show expenses and income as separate dataframe
    #card_expenses = self.iloc[np.where(self['transaction_class'] == "expense")]
    #card_expenses_by_user = self.iloc[np.where(self['transaction_class'] == "expense")].groupby('unique_mem_id').sum()
    #card_income = self.iloc[np.where(self['transaction_class'] == "income")]
    #card_income_by_user = self.iloc[np.where(self['transaction_class'] == "income")].groupby('unique_mem_id').sum()
    #bank_expenses = df_bank.iloc[np.where(df_bank['transaction_class'] == "expense")]
    #bank_expenses_by_user = df_bank.iloc[np.where(df_bank['transaction_class'] == "expense")].groupby('unique_mem_id').sum()
    #bank_income = df_bank.iloc[np.where(df_bank['transaction_class'] == "income")]
    #bank_income_by_user = df_bank.iloc[np.where(df_bank['transaction_class'] == "income")].groupby('unique_mem_id').sum()
    #%%
    #df_1 = self[['unique_mem_id', 'amount', 'transaction_class']].groupby('unique_mem_id')

    for n in id_list:
        df_1 = self[['unique_mem_id', 'amount', 'transaction_class']][self[unique_mem_id == n]]
        print("PANEL INJECTION")
        #open initially and only write to the file to generate the headers
        with open('Card_Panel_Injection.csv', 'w') as newFile:
            newFileWriter=csv.writer(newFile)
            newFileWriter.writerow("Refers to: CARD_PANEL")
            newFileWriter.writerow(["User_ID", "Injection in USD required"])
        # f = open('test.csv', 'w')
        # with f:
            # fnames = ["User_ID", "Injection in USD required"]
            # writer = csv.DictWriter(f, fieldnames=fnames)
            # writer.writeheader()

        #DF_1
        try:
            cumulative_amount = []
            amount_list = []
            for row in df_1.itertuples():
                #access data using column names
                if row.transaction_class == "expense":
                    #print(index, row.unique_mem_id, row.amount, row.transaction_class)
                    amount_list.append(row.amount)
                    cumulative_amount = np.cumsum(amount_list, axis = 0)
                    #print(row.unique_mem_id, cumulative_amount)
                else:
                    #print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
                    break
            #print out the member id as part of the for-loop and and the last element of the list which is the amount to be injected
            print(f"unique_member_ID: {row.unique_mem_id}; initial injection needed in USD: {cumulative_amount[-1]}")

            #open/append income and expense per user_id to a CSV that has been created outside the loop
            #writes all rows inside the iteration loop correctly but without headers now
            with open('Card_Panel_Injection.csv', 'a') as newFile:
                newFileWriter=csv.writer(newFile)
                #write per row to a CSV
                newFileWriter.writerow([row.unique_mem_id, cumulative_amount[-1]])
            ##f = open('test.csv', 'a')
            #with f:
                #field names needed in append mode to know the orders of keys and values
                #fnames = ['User_ID', 'Injection in USD required']
                #writer = csv.DictWriter(f, fieldnames=fnames)
                #writer.writerow({'User_ID' : row.unique_mem_id, 'Injection in USD required': cumulative_amount[-1]})

        except Exception as exc:
            for row in df_1.head(1).itertuples():
                with open('Card_Panel_Injection.csv', 'a') as newFile:
                    newFileWriter=csv.writer(newFile)
                    #write per row to a CSV
                    newFileWriter.writerow([row.unique_mem_id, exc])
                ##f = open('test.csv', 'a')
                #with f:
                    #field names needed in append mode to know the orders of keys and values
                    #fnames = ['User_ID', 'Injection in USD required']
                    #writer = csv.DictWriter(f, fieldnames=fnames)
                    #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

            #print(f"There was a problem with user ID: {card_members[0]}; Error: {exc}")
            print(f"There was a problem with user ID: {row.unique_mem_id}; Error: {exc}")
            pass

    if __name__ == "__main__":
        import sys
        close_connection(int(sys.argv[1]))