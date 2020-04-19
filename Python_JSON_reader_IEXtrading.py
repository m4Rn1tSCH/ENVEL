#Packages
import requests
import json5
import pandas as pd
from bs4 import BeautifulSoup

def pull_data_1m():
	#quote, news and charts
	#url = 'https://api.iextrading.com/1.0/stock/market/batch?symbols=aapl,fb,tsla,aig%2b,msft,goog&types=quote,news,chart&range=1m&last=5'
	#quote and chart
	#url = 'https://api.iextrading.com/1.0/stock/market/batch?symbols=aapl,fb,tsla,aig%2b,msft,goog&types=quote,chart&range=1m&last=5'
	#quote
	#url = 'https://api.iextrading.com/1.0/stock/market/batch?symbols=aapl,fb,tsla,aig%2b,msft,goog&types=quote&range=1m&last=5'
	#Specify url													here you change the symbols								here the intervall
	url = 'https://api.iextrading.com/1.0/stock/market/batch?symbols=aapl,fb,tsla,msft,aig%2b,goog&types=quote,news,chart&range=1m&last=5'

	# Package the request, send the request and catch the response: r
	r = requests.get(url)

	# Extracts the response as html: html_doc
	stock_1m_html = r.text

	# create a BeautifulSoup object from the HTML
	# ATTENTION use arg >>features="lxml"<< to use standard parser for HTML (for use on different systems
	soup = BeautifulSoup(stock_1m_html, features="lxml")
	
	json_stocks = open(stock_1m_html, "r")

	##supposed to create the dictionary but it is still HTML
	#dict_stocks = [row.findAll('td') for row in soup.findAll('tr')]
	#results = { td[0].string: td[1].string for td in tds }
	#print results



	#stock_data = []
	#for label in soup.select('div.itemAttr, td.attrLabels'):
	#    stock_data.append({ label.text.strip(): label.find_next_sibling().text.strip() })

	# Print the data 
	print(soup)

