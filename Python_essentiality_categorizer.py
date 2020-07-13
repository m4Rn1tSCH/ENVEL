# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:01:06 2020

@author: bill-
"""

'''
Purpose of this script is to interact with a csv file/database and produce a dictionary
with unique IDs and corresponding income and expenses in separate dictionaries
'''
#load needed packages
import pandas as pd
import numpy as np
from datetime import datetime as dt
import os
import csv

def categorization(file_path):

    '''
    Datetime engineering for card and bank panel
    These columns help for reporting like weekly or monthly expenses and
    improve prediction of re-occurring transactions
    '''
    for col in list(df_card):
        if df_card[col].dtype == 'datetime64[ns]':
            df_card[f"{col}_month"] = df_card[col].dt.month
            df_card[f"{col}_week"] = df_card[col].dt.week
            df_card[f"{col}_weekday"] = df_card[col].dt.weekday

    # datetime engineering bank_panel
    for col in list(df_bank):
        if df_bank[col].dtype == 'datetime64[ns]':
            df_bank[f"{col}_month"] = df_bank[col].dt.month
            df_bank[f"{col}_week"] = df_bank[col].dt.week
            df_bank[f"{col}_weekday"] = df_bank[col].dt.weekday

    #%%
    '''
    POSTGRESQL COLUMNS - CLASSIFICATION OF TRANSACTIONS
    Following lists contains the categories to classify transactions either as expense or income
    names taken directly from the Yodlee dataset; can be appended at will
    '''
    #essential/ non-essential transactions
    card_ess = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Gifts']
    card_non_ess = ['Groceries', 'Automotive/Fuel', 'Home Improvement', 'Travel',
                'Restaurants', 'Healthcare/Medical', 'Credit Card Payments',
                'Electronics/General Merchandise', 'Entertainment/Recreation',
                'Postage/Shipping', 'Other Expenses', 'Personal/Family',
                'Service Charges/Fees', 'Services/Supplies', 'Utilities',
                'Office Expenses', 'Cable/Satellite/Telecom',
                'Subscriptions/Renewals', 'Insurance']
    bank_ess = ['Deposits', 'Salary/Regular Income', 'Transfers',
                'Investment/Retirement Income', 'Rewards', 'Other Income',
                'Refunds/Adjustments', 'Interest Income', 'Gifts', 'Expense Reimbursement']
    bank_non_ess = ['Service Charges/Fees',
                'Credit Card Payments', 'Utilities', 'Healthcare/Medical', 'Loans',
                'Check Payment', 'Electronics/General Merchandise', 'Groceries',
                'Automotive/Fuel', 'Restaurants', 'Personal/Family',
                'Entertainment/Recreation', 'Services/Supplies', 'Other Expenses',
                'ATM/Cash Withdrawals', 'Cable/Satellite/Telecom',
                'Postage/Shipping', 'Insurance', 'Travel', 'Taxes',
                'Home Improvement', 'Education', 'Charitable Giving',
                'Subscriptions/Renewals', 'Rent', 'Office Expenses', 'Mortgage']
    '''
    Iterate through rows and create new columns with a keyword that it is either an expense or income
    This part is needed to make sure that initial balances can be determined better
    '''
    # card panel
    try:
        transaction_class_card = pd.Series([], dtype = 'object')
        for index, i in enumerate(df_card['transaction_category_name']):
            if i in card_inc:
                transaction_class_card[index] = "essential"
            elif i in card_exp:
                transaction_class_card[index] = "non_essential"
            else:
                transaction_class_card[index] = "unknown"
        df_card.insert(loc = len(df_card.columns), column = "t_essentiality", value = transaction_class_card)
    except:
        print("column is already existing or  error")

    # bank_panel
    try:
        transaction_class_bank = pd.Series([], dtype = 'object')
        for index, i in enumerate(df_bank['transaction_category_name']):
            if i in bank_inc:
                transaction_class_bank[index] = "essential"
            elif i in bank_exp:
                transaction_class_bank[index] = "non_essential"
            else:
                transaction_class_bank[index] = "unknown"
        df_bank.insert(loc = len(df_bank.columns), column = "t_essentiality", value = transaction_class_bank)
    except:
        print("Column is already existing or error")
#%%
    '''
    POSTGRE-SQL COLUMNS - ALLOCATION TO ENVELOPES
    This section adds a classification of transaction categories to allow a proper allocation to either the cash or the bills envelope
    Bills describes as of 3/26/2020 all kinds of payment whose occurrence is beyond one's control,
    that comes due and for which non-compliance has severe consequences
    All other kinds of payments that are of optional nature and can be avoided are classifed as cash
    '''
    cash_env_card = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Gifts',
                     'Restaurants', 'Electronics/General Merchandise',
                     'Entertainment/Recreation', 'Postage/Shipping', 'Other Expenses',
                     'Personal/Family','Groceries', 'Automotive/Fuel',  'Travel']

    bill_env_card = ['Home Improvement', 'Healthcare/Medical', 'Credit Card Payments'
                     'Service Charges/Fees', 'Services/Supplies', 'Utilities',
                     'Office Expenses', 'Cable/Satellite/Telecom',
                     'Subscriptions/Renewals', 'Insurance']

    cash_env_bank = ['Deposits', 'Salary/Regular Income', 'Transfers',
                     'Investment/Retirement Income', 'Rewards', 'Other Income',
                     'Refunds/Adjustments', 'Interest Income', 'Gifts', 'Expense Reimbursement',
                     'Electronics/General Merchandise', 'Groceries', 'Automotive/Fuel',
                     'Restaurants', 'Personal/Family', 'Entertainment/Recreation',
                     'Services/Supplies', 'Other Expenses', 'ATM/Cash Withdrawals',
                     'Postage/Shipping', 'Travel', 'Education', 'Charitable Giving',
                     'Office Expenses']

    bill_env_bank = ['Service Charges/Fees', 'Credit Card Payments',
                     'Utilities', 'Healthcare/Medical', 'Loans', 'Check Payment',
                     'Cable/Satellite/Telecom', 'Insurance', 'Taxes', 'Home Improvement',
                     'Subscriptions/Renewals', 'Rent', 'Mortgage']
    #iterate through rows and create a new columns with a note that it is either an expense or income
    # card panel
    try:
        envelope_cat_card = pd.Series([], dtype = 'object')
        for index, i in enumerate(df_card['transaction_category_name']):
            if i in cash_env_card:
                envelope_cat_card[index] = "cash"
            elif i in bill_env_card:
                envelope_cat_card[index] = "bill"
            else:
                envelope_cat_card[index] = "NOT_CLASSIFIED"
        df_card.insert(loc = len(df_card.columns), column = "envelope_category", value = envelope_cat_card)
    except:
        print("CASH/BILL column is already existing or another error")

    # bank_panel
    try:
        envelope_cat_bank = pd.Series([], dtype = 'object')
        for index, i in enumerate(df_bank['transaction_category_name']):
            if i in cash_env_bank:
                envelope_cat_bank[index] = "cash"
            elif i in bill_env_bank:
                envelope_cat_bank[index] = "bill"
            else:
                envelope_cat_bank[index] = "NOT_CLASSIFIED"
        df_bank.insert(loc = len(df_bank.columns), column = "envelope_category", value = envelope_cat_bank)
    except:
        print("CASH/BILL column is already existing or another error")
    #%%
    '''
    Filter for dataframes to find out income and expenses narrowed down to the user id
    '''
    #filter with ilocation and show expenses and income as spearate dataframe
    card_expenses = df_card.iloc[np.where(df_card['transaction_class'] == "expense")]
    card_expenses_by_user = df_card.iloc[np.where(df_card['transaction_class'] == "expense")].groupby('unique_mem_id').sum()
    card_income = df_card.iloc[np.where(df_card['transaction_class'] == "income")]
    card_income_by_user = df_card.iloc[np.where(df_card['transaction_class'] == "income")].groupby('unique_mem_id').sum()
    bank_expenses = df_bank.iloc[np.where(df_bank['transaction_class'] == "expense")]
    bank_expenses_by_user = df_bank.iloc[np.where(df_bank['transaction_class'] == "expense")].groupby('unique_mem_id').sum()
    bank_income = df_bank.iloc[np.where(df_bank['transaction_class'] == "income")]
    bank_income_by_user = df_bank.iloc[np.where(df_bank['transaction_class'] == "income")].groupby('unique_mem_id').sum()
    #%%
    '''
    POSTGRESQL - BUDGET SUGGESTION COLUMNS
    Create columns with an initial recommendation of the budgeting mode and the corresponding daily limit
    Logic is based on the weekly or biweekly income:
    Logic of stability of spending behavior and standard deviation within various time frames
    Behavior is considered: stable and non-erratic when:
        LATER:Std dev of past 3 days is still smaller than emergency cash allocated for a day
        LATER:Std dev of past week is still smaller than emergency allocated for a week
        LATER:Std dev of 30d is smaller than 70% of monthly income
        (to allow purchase of flight tickets or hotel stays without forcing a change of the spending mode)
    '''
    # card panel
    #try:
    print("CARD PANEL BUDGETING REPORT")
    budget_mode_card = pd.Series([], dtype = 'object')
    for index, i, e, c in zip(bank_income_by_user.index, bank_income_by_user.amount,
                              bank_expenses_by_user.amount, card_expenses_by_user.amount):
        if i > e + c:
            print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Good Budget!")
            budget_mode_card[index] = "normal mode"
        else:
            print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Overspending!")
            budget_mode_card[index] = "beastmode"
    df_card.insert(loc = len(df_card.columns), column = "budget_mode_suggestion_card", value = budget_mode_card)
    #except:
        #print("values overwritten in card panel")
    # bank_panel
    #try:
    budget_mode_bank = pd.Series([], dtype = 'object')
    print("BANK PANEL BUDGETING REPORT")
    for index, i, e, c in zip(bank_income_by_user.index, bank_income_by_user.amount,
                              bank_expenses_by_user.amount, card_expenses_by_user.amount):
        if i > e + c:
            print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Good Budget!")
            budget_mode_bank[index] = "normal mode"
        else:
            print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Overspending!")
            budget_mode_bank[index] = "beastmode"
    df_bank.insert(loc = len(df_bank.columns), column = "budget_mode_suggestion_card", value = budget_mode_bank)
    #except:
        #print("values overwritten in bank panel")
    #%%
    df_card.set_index("optimized_transaction_date", drop = False, inplace = True)
    df_bank.set_index("optimized_transaction_date", drop = False, inplace = True)
    #%%
    '''
    IMPROVISED SOLUTION WITHOUT ITERATION
    Filter-df by unique id of each customer with columns: member_id; amount; envelope_category; transaction_class
    iteration over each row as tuples and append amount to a list.
    This list is taken and used for a cumulative sum of all transactions with type "expense"
    Until "income" class is hit to stop
    Numerical amount needs to be injected for simulation
    problem of Python here; one cannot assign an element to a list that is not yet existing
    '''
    df_1 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '70850441974905670928446']
    df_2 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '201492116860211330700059']
    df_3 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '257154737161372702866152']
    df_4 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '364987015290224198196263']
    df_5 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '651451454569880463282551']
    df_6 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '748150568877494117414131']
    #%%
    print("CARD PANEL INJECTION")
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

        print(f"There was a problem with user ID: {card_members[0]}; Error: {exc}")
        pass

    ##DF_2
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_2.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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

        print(f"There was a problem with user ID: {card_members[1]}; Error: {exc}")
        pass

    ##DF_3
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_3.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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

        print(f"There was a problem with user ID: {card_members[2]}; Error: {exc}")
        pass

    ##DF_4
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_4.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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

        print(f"There was a problem with user ID: {card_members[3]}; Error: {exc}")
        pass

    ##DF_5
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_5.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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

        print(f"There was a problem with user ID: {card_members[4]}; Error: {exc}")
        pass

    ##DF_6
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_6.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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

        print(f"There was a problem with user ID: {card_members[5]}; Error: {exc}")
        pass
    #%%
    df_70850441974905670928446 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '70850441974905670928446']
    df_257154737161372702866152 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '257154737161372702866152']
    df_364987015290224198196263 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '364987015290224198196263']
    df_579758724513140495207829 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '579758724513140495207829']
    df_630323465162087035360618 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '630323465162087035360618']
    df_635337295180631420039874 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '635337295180631420039874']
    df_1187627404526562698645364 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '1187627404526562698645364']
    #%%
    print("BANK PANEL INJECTION")
    #open initially and only write to the file to generate the headers
    with open('Bank_Panel_Injection.csv', 'w') as newFile:
        newFileWriter=csv.writer(newFile)
        newFileWriter.writerow(["Refers to:", "BANK_PANEL"])
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
        for row in df_70850441974905670928446.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print(index, row.unique_mem_id, row.amount, row.transaction_class)
                amount_list.append(row.amount)
                cumulative_amount = np.cumsum(amount_list, axis = 0)
                print(row.unique_mem_id, cumulative_amount)
            else:
                #print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
                break
        #print out the member id as part of the for-loop and and the last element of the list which is the amount to be injected
        print(f"unique_member_ID: {row.unique_mem_id}; initial injection needed in USD: {cumulative_amount[-1]}")
        #open/append income and expense per user_id to a CSV that has been created outside the loop
        #writes all rows inside the iteration loop correctly but without headers now
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        ##f = open('test.csv', 'a')
        #with f:
            #field names needed in append mode to know the orders of keys and values
            #fnames = ['User_ID', 'Injection in USD required']
            #writer = csv.DictWriter(f, fieldnames=fnames)
            #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})
        print(f"There was a problem with user ID: {bank_members[0]}; Error: {exc}")
        pass

    ##DF_2
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_257154737161372702866152.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        ##f = open('test.csv', 'a')
        #with f:
            #field names needed in append mode to know the orders of keys and values
            #fnames = ['User_ID', 'Injection in USD required']
            #writer = csv.DictWriter(f, fieldnames=fnames)
            #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[1]}; Error: {exc}")
        pass

    ##DF_3
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_364987015290224198196263.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        # ##f = open('test.csv', 'a')
        # #with f:
        #     #field names needed in append mode to know the orders of keys and values
        #     #fnames = ['User_ID', 'Injection in USD required']
        #     #writer = csv.DictWriter(f, fieldnames=fnames)
        #     #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[2]}; Error: {exc}")
        pass

    ##DF_4
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_579758724513140495207829.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        # ##f = open('test.csv', 'a')
        # #with f:
        #     #field names needed in append mode to know the orders of keys and values
        #     #fnames = ['User_ID', 'Injection in USD required']
        #     #writer = csv.DictWriter(f, fieldnames=fnames)
        #     #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[3]}; Error: {exc}")
        pass

    ##DF_5
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_630323465162087035360618.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
                amount_list.append(row.amount)
                cumulative_amount = np.cumsum(amount_list, axis = 0)
                #print(row.unique_mem_id, cumulative_amount)
            else:
               # print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
                break
        #print out the member id as part of the for-loop and and the last element of the list which is the amount to be injected
        print(f"unique_member_ID: {row.unique_mem_id}; initial injection needed in USD: {cumulative_amount[-1]}")

        #open/append income and expense per user_id to a CSV that has been created outside the loop
        #writes all rows inside the iteration loop correctly but without headers now
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, cumulative_amount[-1]])
        ##f = open('test.csv', 'a')
        #with f:
            #field names needed in append mode to know the orders of keys and values
            #fnames = ['User_ID', 'Injection in USD required']
            #writer = csv.DictWriter(f, fieldnames=fnames)
            #writer.writerow({'User_ID' : row.unique_mem_id, 'Injection in USD required': cumulative_amount[-1]})

    except  Exception as exc:

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        # ##f = open('test.csv', 'a')
        # #with f:
        #     #field names needed in append mode to know the orders of keys and values
        #     #fnames = ['User_ID', 'Injection in USD required']
        #     #writer = csv.DictWriter(f, fieldnames=fnames)
        #     #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[4]}; Error: {exc}")
        pass

    ##DF_6
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_635337295180631420039874.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        # ##f = open('test.csv', 'a')
        # #with f:
        #     #field names needed in append mode to know the orders of keys and values
        #     #fnames = ['User_ID', 'Injection in USD required']
        #     #writer = csv.DictWriter(f, fieldnames=fnames)
        #     #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[5]}; Error: {exc}")
        pass

    ##DF_7
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_1187627404526562698645364.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        # ##f = open('test.csv', 'a')
        # #with f:
        #     #field names needed in append mode to know the orders of keys and values
        #     #fnames = ['User_ID', 'Injection in USD required']
        #     #writer = csv.DictWriter(f, fieldnames=fnames)
        #     #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[6]}; Error: {exc}")
        pass
#%%
    '''
    CHART FOR EACH USER'S INCOME, EXPENSES AND EXCESS MONEY
    The loop uses the filtered dataframes which are narrowed down by user and
    show the budgeting ability of unique user ID found in the panel
    '''
    #index = index
    #i = income
    #e = expense
    '''
    REPORTING CSV - YODLEE DATA
    Write it on a per-line basis to the csv that will either sit sit in the flask folder
    or can be saved in the current working directory and will deliver information for the disconnected injector
    '''
    try:
        #open initially and only write to the file to generate the headers
        with open('User_ID_transactions.csv','w') as newFile:
            newFileWriter=csv.writer(newFile)
            newFileWriter.writerow(["User_ID", "Income", "Expenses", "Excess_Cash(+)/Debt(-)"])
        # f = open('test.csv', 'w')
        # with f:
            # fnames = ['User_ID', 'income', 'expense', 'difference']
            # writer = csv.DictWriter(f, fieldnames=fnames)
            # writer.writeheader()
        for index, i, e in zip(bank_income_by_user.index, bank_income_by_user.amount, bank_expenses_by_user.amount):
            if i > e:
                print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Good Budget!; Excess cash: {i - e}")
            else:
                print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Overspending!; Debt: {i - e}")
            #open/append income and expense per user_id to a CSV that has been created outside the loop
            #writes all rows inside the iteration loop correctly but without headers now
            with open('User_ID_transactions.csv','a') as newFile:
                newFileWriter=csv.writer(newFile)
                #write per row to a CSV
                newFileWriter.writerow([index, i, e, i - e])
            ##f = open('test.csv', 'a')
            #with f:
                #field names needed in append mode to know the orders of keys and values
                #fnames = ['User_ID', 'income', 'expense', 'difference']
                #writer = csv.DictWriter(f, fieldnames=fnames)
                #writer.writerow({'User_ID' : index, 'income': i, 'expense': e, 'difference': i-e})
    except:
        print("data by user might not be available; check the number of unique user IDs")

    return 'Preprocessing completed.'