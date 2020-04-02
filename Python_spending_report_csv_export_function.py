# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:21:10 2020

@author: bill-
"""

'''
This module conducts calculations per user ID
determines:
    daily spending (mean/std)
    weekly spending (mean/std)
    monthly spending (mean/std)
the report is saved as a pandas df and converted to a CSV
'''
#in flask body with variable input
#allows to input file
def spending_report(self):

    test_path = r'C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx'
    #relative path to test the file sitting directly in the folder with the script
    #test_path_2 = './2020-01-28 envel.ai Working Class Sample.xlsx'

    df_card = pd.read_excel(os.path.abspath(test_path), sheet_name = "Card Panel")
    card_members = df_card['unique_mem_id'].unique()
         #%%
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
#%%
    '''
    Addition of feature columns for additive spending on a weekly; monthly; daily basis
    These dataframes are then convertable to a CSV for reporting purposes or could be shown in the app
    As of 4/2/2020 the spending report generates a file-wide dataframe based on all users together
    '''
    #total throughput of money
    total_throughput = df_card['amount'].sum()
    #monthly figures
    net_monthly_throughput = df_card['amount'].groupby(df_card['transaction_date_month']).sum()
    avg_monthly_throughput = df_card['amount'].groupby(df_card['transaction_date_month']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    monthly_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_week']).sum()
    #monthly_expenses = df_card['amount'][df_card['transaction_base_type'] == 'debit'].groupby(df_card['transaction_date_week']).sum()
    #weekly figures
    net_weekly_throughput = df_card['amount'].groupby(df_card['transaction_date_week']).sum()
    avg_weekly_throughput = df_card['amount'].groupby(df_card['transaction_date_week']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    weekly_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_week']).sum()
    #weekly_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_week']).sum()
    #daily figures
    net_daily_spending = df_card['amount'].groupby(df_card['transaction_date_weekday']).sum()
    avg_daily_spending = df_card['amount'].groupby(df_card['transaction_date_weekday']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    daily_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_weekday']).sum()
    #daily_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_weekday']).sum()

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




        return 'Spending report generated; CSV-file in current working directory.'