from psycopg2 import OperationalError
import numpy as np
import pandas as pd
from datetime import datetime as dt

# FILE IMPORTS FOR NOTEBOOKS
from SQL_connection import execute_read_query, create_connection
import PostgreSQL_credentials as acc

def users_df(section=1):
    connection = create_connection(db_name=acc.YDB_name,
                                   db_user=acc.YDB_user,
                                   db_password=acc.YDB_password,
                                   db_host=acc.YDB_host,
                                   db_port=acc.YDB_port)

    fields = ['unique_mem_id', 'amount', 'transaction_base_type',
              'transaction_category_name', 'optimized_transaction_date']

    data = []
    columns = ['unique_mem_id', 'monthly_income',
               'monthly_savings', 'monthly_savings_envel']
    
    # number of days in month for 2019 (yodlee db year)
    days_monthly = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # generator statement that is exhausted and then deleted
    transaction_month = [x for x in range(1, 13)]
    # transaction_categorical_name to include as bills
    bills_categories = ['Cable/Satellite/Telecom', 'Healthcare/Medical', 'Insurance',
                       'Mortgage', 'Rent', 'Subscriptions/Renewals', 'Utilities',
                       'Loans', 'Education']

    try:
        filter_query = f"select unique_mem_id from user_demographic order by unique_mem_id asc"
        users = execute_read_query(connection, filter_query)
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        connection.rollback
    # split up the number of total users into an 10 equal arrays
    users_sections = np.split(np.array(users), 10)

    for num, user in enumerate(users_sections[section - 1]):
        try:
            filter_query = f"(SELECT {', '.join(field for field in fields)} \
                                FROM card_record WHERE unique_mem_id = '{user[0]}') \
                                UNION (SELECT {', '.join(field for field in fields)} \
                                       FROM bank_record WHERE unique_mem_id = '{user[0]}')"
            transaction_query = execute_read_query(connection, filter_query)
            df = pd.DataFrame(transaction_query, columns=fields)
            print(f"{len(df)} transactions for user {user[0]}, {num+1}/10000 users, PROGRESS: {round(((num+1)/10000)*100, 2)}%.")
        except OperationalError as e:
            print(f"The error '{e}' occurred")
            connection.rollback

        df['amount'] = df['amount'].astype('float64')
        # add date columns
        df['optimized_transaction_date'] = pd.to_datetime(
            df['optimized_transaction_date'])
        df["transaction_month"] = df['optimized_transaction_date'].dt.month
        df["transaction_day"] = df['optimized_transaction_date'].dt.day

        monthly_income = df['amount'][df['transaction_base_type'].eq('credit')].groupby(
            df['transaction_month']).sum().round(2)
        # fill all 12 months as some users missing data
        for i in transaction_month:
            try:
                monthly_income[i]
            except:
                monthly_income[i] = 0
        monthly_expenses = df['amount'][df['transaction_base_type'].eq('debit')].groupby(
            df['transaction_month']).sum().round(2)
        monthly_savings = round(monthly_income - monthly_expenses, 2)

        # total bills per month according to categories, then remove bills transactions from df
        monthly_bills = df['amount'][df['transaction_category_name'].isin(bills_categories)].groupby(
            df['transaction_month']).sum().round(2)
        for i in transaction_month:
            try:
                monthly_bills[i]
            except:
                monthly_bills[i] = 0
        df = df[~df['transaction_category_name'].isin(bills_categories)]

        # ai devisions
        monthly_emergency = round(monthly_income * 0.1, 2)
        monthly_vault = round(monthly_income * 0.1, 2)
        monthly_cash = round(
            monthly_income - (monthly_bills * 1.1) - monthly_emergency - monthly_vault, 2)
        monthly_daily = round(monthly_cash / days_monthly, 2)

        # calculations if ai mode was on
        daily_overspent = []
        daily_underspent = []
        for month, days in enumerate(days_monthly, start=1):
            daily_cash = monthly_daily[month]
            daily_total_underspent = 0
            daily_total_overspent = 0
            daily_expenses = df['amount'][df['transaction_base_type'].eq('debit') & df['transaction_month'].eq(month)].groupby(
                df['transaction_day']).sum()
            for day in range(1, days + 1):
                try:
                    daily_expense = daily_expenses[day]
                except:
                    daily_expense = 0
                daily_left = round(daily_cash - daily_expense, 2)
                if daily_left > 0:
                    daily_total_underspent = daily_total_underspent + daily_left
                else:
                    daily_total_overspent = daily_total_overspent - daily_left
            daily_overspent.append(round(daily_total_overspent, 2))
            daily_underspent.append(round(daily_total_underspent, 2))

        # calculations for nett benefit from ai
        monthly_vault_end = monthly_vault - daily_overspent
        monthly_emergency_end = monthly_emergency + daily_underspent
        for i, vault in enumerate(monthly_vault_end, start=1):
            if vault > 0:
                monthly_emergency_end[i] = monthly_emergency_end[i] + vault
        monthly_savings_envel = monthly_emergency_end + abs(monthly_vault_end)

        unique_mem_id = user[0]
        for i in transaction_month:
            try:
                data.append([unique_mem_id, monthly_income[i], monthly_savings[i], monthly_savings_envel[i]])
            except:
              print(f"missing values for month {i} of user {unique_mem_id}")

    trans_df = pd.DataFrame(data, columns=columns)

    # calculations for results using numpy (faster than pd)
    avg_savings_pu = np.array(trans_df.groupby('unique_mem_id').mean()['monthly_savings'])
    avg_savings_pupm = avg_savings_pu.mean()
    avg_savings_pu_envel = np.array(trans_df.groupby('unique_mem_id').mean()['monthly_savings_envel'])
    avg_savings_pupm_envel = avg_savings_pu_envel.mean()

    aggr_savings = np.array(trans_df['monthly_savings']).sum()
    aggr_savings_envel = np.array(trans_df['monthly_savings_envel']).sum()

    avr_inc_pu = np.array(trans_df.groupby('unique_mem_id').mean()['monthly_income'])
    num_living_ptp = (avg_savings_pu < avr_inc_pu).sum()
    num_living_ptp_envel = (avg_savings_pu_envel < avr_inc_pu).sum()

    results = {'avg_savings_pupm': avg_savings_pupm,
               'avg_savings_pupm_envel': avg_savings_pupm_envel,
               'aggr_savings': aggr_savings,
               'aggr_savings_envel': aggr_savings_envel,
               'num_living_ptp': num_living_ptp,
               'num_living_ptp_envel': num_living_ptp_envel}

    return results

    # return trans_df
