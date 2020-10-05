#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:59:42 2020

@author: bill
"""


import requests, os
import urllib.request
import time
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime as dt

def save_data(filename,data):
	raw = os.getcwd()
	date_of_creation = dt.today().strftime('%d%m%Y%H%M')
	csv_path = os.path.abspath(os.path.join(raw + date_of_creation + '_' + filename + '.csv'))
	data.to_csv(csv_path)
	print(f"{filename}.csv saved to {csv_path}")

states = ['ohio','illinois','kansas','northdakota','oklahoma','wisconsin','missouri','newhampshire','texas','alaska',
'newmexico','maryland','iowa','colorado','delaware','kentucky','southcarolina','nevada','hawaii','montana',
'wyoming','michigan','washington','alabama','arizona','northcarolina','mississippi','tennessee','louisiana','massachusetts',
'newjersey','southdakota','arkansas','connecticut','georgia','pennsylvania','florida','districtofcolumbia','nebraska','maine',
'california','indiana','newyork','minnesota','idaho','utah','westvirginia','vermont','oregon','rhodeisland','virginia']


all_cities = []

for state in states:
	URL = ''.join('https://www.citypopulation.de/en/usa/cities/'+state+'/')
	response = requests.get(URL)
	print(f'State: {state} giving response: {response}')

	soup = BeautifulSoup(response.text, 'html.parser')

	cities = []

	for d in soup.findAll('td', attrs={'class':'rname'}):
		cities.append(d.string)

	cities.pop(0)
	print(f"Total in Cities in {state} : {len(cities)}")
	all_cities += cities
	time.sleep(1)


print(all_cities)
print(len(all_cities))
save_data('web_cities', pd.DataFrame(all_cities))

