#Packages
import os
import pandas as pd

#get input link
link = r"C:\Users\bill-\Desktop\TransactionsD.csv"
new_link = link.replace(os.sep, '/')
input_dir = ''.join(('', new_link,''))

#list
bill_list = ['Amazon', 'Netflix', 'AMZN', 'NATIONALGRID', 'Eversource', 'Xfinity', 'Comcast']

file_in = pd.read_csv(input_dir)
items = file_in.items()

for label, content in items:
        print('label:', label)
        print('content:', content, sep='\n')


file_in[file_in['ShopName'].str.contains('Amazon')]

file_in[file_in['ShopName'].str.contains('Wine')]