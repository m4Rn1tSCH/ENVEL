# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:22:31 2020

@author: bill-
"""

#required packages
import os
from datetime import datetime as dt

'''
CONVERSION TO CSV
This function converts a dataframe to CSV and just work in appending mode
if the csv file is already existing.
Saves with a timestamp and generates in the current working directory
'''
def csv_export(df, file_name):
    #local working directory
    raw = os.getcwd()
    #folder when executed on the AWS instance
    #aws = os.mkdir('/injection')
    date_of_creation = dt.today().strftime('%m-%d-%Y_%Hh-%mmin')

    csv_path = os.path.abspath(os.path.join(raw, date_of_creation + '_' + file_name + '_REPORT' + '.csv'))

    try:
        df.to_csv(csv_path)
    except FileExistsError as exc:
        print(exc)
        print("existing file will be appended instead...")
        csv_path = os.path.abspath(os.path.join(raw, date_of_creation + '_' + file_name + '_REPORT' + '.csv'))
        df.to_csv(csv_path, mode = 'a', header = False)

#close the function with return xx to avoid error 500 when querying the URL and have a message showing up instead
    return 'Spending report generated; CSV-file in current working directory.'

#add this part at the end to make the module executable as script
#takes arguments here (df)
#
    if __name__ == "__main__":
        import sys
        csv_export(int(sys.argv[1]))