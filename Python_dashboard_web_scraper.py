#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:17:11 2018
@author: kerry
"""

#Import libraries
import urllib.request
from bs4 import BeautifulSoup
import csv
#%%

#Specify the URL
urlpage = 'http://www.fasttrack.co.uk/league-tables/tech-track-100/league-table/'
print(urlpage)
#Query the website and return the HTML to the variable 'page'
page = urllib.request.urlopen(urlpage)
#Parse the HTML using beautiful soup and store in variable 'soup'
soup = BeautifulSoup(page, 'html.parser')
#Find results within table
table = soup.find('table', attrs={'class': 'tableSorter'})
results = table.find_all('tr')
print('Number of results', len(results))

#Create and write headers to a list
rows = []
rows.append(['Rank', 'Company Name', 'Webpage', 'Description', 'Location', 'Year end', 'Annual sales rise over 3 years', 'Sales £000s', 'Staff', 'Comments'])

#Loop over results
for result in results:
    #Find all columns per result
    data = result.find_all('td')
    #Check that columns have data
    if len(data) == 0:
        continue

    #Write columns to variables
    rank = data[0].getText()
    company = data[1].getText()
    location = data[2].getText()
    yearend = data[3].getText()
    salesrise = data[4].getText()
    sales = data[5].getText()
    staff = data[6].getText()
    comments = data[7].getText()

    #Print('Company is', company)
    #Company is WonderblyPersonalised children's books
    #Print('Sales', sales)
    #Sales *25,860

    #Extract description from the name
    companyname = data[1].find('span', attrs={'class':'company-name'}).getText()
    description = company.replace(companyname, '')

    #Remove unwanted characters
    sales = sales.strip('*').strip('†').replace(',','')

    #Go to link and extract company website
    url = data[1].find('a').get('href')
    page = urllib.request.urlopen(url)
    #Parse the html using beautiful soup and store in variable 'soup'
    soup = BeautifulSoup(page, 'html.parser')
    #Find the last result in the table and get the link
    try:
        tableRow = soup.find('table').find_all('tr')[-1]
        webpage = tableRow.find('a').get('href')
    except:
        webpage = None

    #Write each result to rows
    rows.append([rank, companyname, webpage, description, location, yearend, salesrise, sales, staff, comments])

print(rows)


##Create csv and write rows to output file
with open('techtrack100.csv','w', newline='') as f_output:
    csv_output = csv.writer(f_output)
    csv_output.writerows(rows)
