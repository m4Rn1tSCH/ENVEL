# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:21:10 2020

@author: bill-
"""

'''
This module conducts calculations if fed with a dataframe from Yodlee
determines:
    daily spending (mean/std)
    weekly spending (mean/std)
    monthly spending (mean/std)
the report is saved as a pandas df and converted to a CSV
the CSV is saved in the current working directory of the device
'''
import os
from datetime import datetime as dt
import pandas as pd

#in flask body with variable input
#allows to input file
#self in python: self is updating an instance variable of its own function
#in this case thei nstance is the dataframe fed to the method and that is upposed to be processed

def spending_report(self):
    #%%
    #temporary test section
#    test_path = r'C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx'
    #relative path to test the file sitting directly in the folder with the script
    #test_path_2 = './2020-01-28 envel.ai Working Class Sample.xlsx'

#    df_card = pd.read_excel(os.path.abspath(test_path), sheet_name = "Card Panel")
#    card_members = df_card['unique_mem_id'].unique()
    #%%
    '''
    Datetime engineering for card and bank panel
    These columns help for reporting like weekly or monthly expenses and
    improve prediction of re-occurring transactions
    '''
    for col in list(self):
        if self[col].dtype == 'datetime64[ns]':
            self[f"{col}_month"] = self[col].dt.month
            self[f"{col}_week"] = self[col].dt.week
            self[f"{col}_weekday"] = self[col].dt.weekday

    '''
    Addition of feature columns for additive spending on a weekly; monthly; daily basis
    These dataframes are then convertable to a CSV for reporting purposes or could be shown in the app
    As of 4/2/2020 the spending report generates a file-wide dataframe based on all users together
    '''
    #total throughput of money
    total_throughput = self['amount'].sum()
    #monthly figures
    net_monthly_throughput = self['amount'].groupby(self['transaction_date_month']).sum()
    avg_monthly_throughput = self['amount'].groupby(self['transaction_date_month']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    monthly_gain = self['amount'][self['amount'] >= 0].groupby(self['transaction_date_week']).sum()
    #monthly_expenses = self['amount'][self['transaction_base_type'] == 'debit'].groupby(self['transaction_date_week']).sum()
    #weekly figures
    net_weekly_throughput = self['amount'].groupby(self['transaction_date_week']).sum()
    avg_weekly_throughput = self['amount'].groupby(self['transaction_date_week']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    weekly_gain = self['amount'][self['amount'] >= 0].groupby(self['transaction_date_week']).sum()
    #weekly_expenses = self['amount'][self['transaction_base_type'] == "debit"].groupby(self['transaction_date_week']).sum()
    #daily figures
    net_daily_spending = self['amount'].groupby(self['transaction_date_weekday']).sum()
    avg_daily_spending = self['amount'].groupby(self['transaction_date_weekday']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    daily_gain = self['amount'][self['amount'] >= 0].groupby(self['transaction_date_weekday']).sum()
    #daily_expenses = self['amount'][self['transaction_base_type'] == "debit"].groupby(self['transaction_date_weekday']).sum()

    #report for users about their spending patterns, given in various intervals
    try:
        print(f"The total turnover on your account has been ${total_throughput}")
        print("................................................................")
        spending_metrics_monthly = pd.DataFrame(data = {'Average Monthly Spending':avg_monthly_throughput,
                                                        'Monthly Turnover':net_monthly_throughput})
        print(spending_metrics_monthly)
        print(".................................................................")
        spending_metrics_weekly = pd.DataFrame(data = {'Average Weekly Spending':avg_weekly_throughput,
                                                       'Weekly Turnover':net_weekly_throughput})
        print(spending_metrics_weekly)
        print(".................................................................")
        spending_metrics_daily = pd.DataFrame(data = {'Average Daily Spending':avg_daily_spending,
                                                      'Daily Turnover':net_daily_spending})
        print(spending_metrics_daily)
    except:
        print("You do not have enough transactions yet. But we are getting there...")

    '''
            CONVERSION OF THE SPENDING REPORTS - ALL USERS
    For testing purposes which does not include randomized IDs as part of the name and allows loading a constant name
    calculations are incorporating all users simultaneously!
    '''
    raw = os.getcwd()
    date_of_creation = dt.today().strftime('%m-%d-%Y_%Hh-%mmin')

    csv_path_msp = os.path.abspath(os.path.join(raw, date_of_creation + '_MONTHLY_REPORT_ALL_USERS' + '.csv'))
    csv_path_wsp = os.path.abspath(os.path.join(raw, date_of_creation + '_WEEKLY_REPORT_ALL_USERS' + '.csv'))
    csv_path_dsp = os.path.abspath(os.path.join(raw, date_of_creation + '_DAILY_REPORT_ALL_USERS' + '.csv'))

    try:
        spending_metrics_monthly.to_csv(csv_path_msp)
        spending_metrics_weekly.to_csv(csv_path_wsp)
        spending_metrics_daily.to_csv(csv_path_dsp)
    except FileExistsError as exc:
        print(exc)
        print("existing file will be appended instead...")
        spending_metrics_monthly.to_csv(csv_path_msp, mode = 'a', header = False)
        spending_metrics_weekly.to_csv(csv_path_wsp, mode = 'a', header = False)
        spending_metrics_daily.to_csv(csv_path_dsp, mode = 'a', header = False)

#close the function with return xx to avoid error 500 when querying the URL and have a message showing up instead
    return 'Spending report generated; CSV-file in current working directory.'

#add this part at the end to make the module executable as script
#takes arguments here (self)
#
    if __name__ == "__main__":
        import sys
        spending_report(int(sys.argv[1]))