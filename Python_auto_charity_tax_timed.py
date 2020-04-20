#!/usr/bin/env python
# coding: utf-8
# Packages
import pandas as pd
from datetime import datetime
import os
#%%
# converting file link
print("file directories will be converted to remove >\< and letter strings")
link = input("path:")
# convert backslashes to slashes
new_link = link.replace(os.sep, '/')
# remove those pesky letter strings
new_link_2 = new_link.replace('"', '')
file_in = ''.join(('', new_link_2, ''))
#%%
# Importing data
data = pd.read_csv(file_in, delimiter = ',')
for i in range(len(data)):
    data['Day of the month'] = data['Day of the month'].asdatetime()
# converting date of month to proper date
data['Day of the month'] = data['Day of the month'].asdatetime
# iterate through all rows until the end of the data frame
for i in range(len(data)):
    data['Day of the month'][i] = datetime.datetime(datetime.datetime.now().year, datetime.datetime.now().month, data['Day of the month'][i])

print(data)
#%%
# setting up lists of taxable classes
meat = ['burgers', 'steak', 'chicken', 'tuna']
alcohol = ['beer', 'wine', 'scotch']
tobacco = ['malboro', 'camel']


# The value of the tax on each type would be determined by the challenge the user opts into
# Its followed by a start and end date
meat_tax = [0.13, '1/10/2019', '30/10/2019']
alcohol_tax = [0.2, '1/7/2019', '31/8/2019']
tobacco_tax = [0.5, '1/8/2019', '31/12/2050']

#%%
## we check to see which transactions took place during the time that each of these challenges was taking place
# check if its still within the time frame of the tax period
meat_tax_timing = (pd.to_datetime(meat_tax[1]) <= data['Day of the month']) & (pd.to_datetime(meat_tax[2]) >= data['Day of the month'])
alcohol_tax_timing = (pd.to_datetime(alcohol_tax[1]) <= data['Day of the month'])&(pd.to_datetime(alcohol_tax[2]) >= data['Day of the month'])
tobacco_tax_timing = (pd.to_datetime(tobacco_tax[1]) <= data['Day of the month'])&(pd.to_datetime(tobacco_tax[2]) >= data['Day of the month'])

#%%
# we determine how much charity you need to give for each cause
meat_monthly_tax = sum(data[meat_tax_timing & data['Item'].isin(meat)]['Cost']) * meat_tax[0]
alcohol_monthly_tax = sum(data[alcohol_tax_timing & data['Item'].isin(alcohol)]['Cost']) * alcohol_tax[0]
tobacco_monthly_tax = sum(data[tobacco_tax_timing & data['Item'].isin(tobacco)]['Cost']) * tobacco_tax[0]

print(meat_monthly_tax, "has been paid to related charities")
print(alcohol_monthly_tax, "has been paid to related charities")
print(tobacco_monthly_tax, "has been paid to related charities")
