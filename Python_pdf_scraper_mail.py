# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:39:28 2020

@author: bill-
"""
#installed is pdfminer.six but is loaded as pdfminer
#internal pakcages seems to be the same
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import re
import os
#%%
def get_cv_email(self, cv_path):
    pagenums = set()
    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)
    infile = open(cv_path, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = match.group(0)
    return email

#\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)
#%%
#set csv_path
basepath = 'C:/Users/bill-/Desktop/Harvard_Resumes'
path_list = []
#error_list = []
#Walking a directory tree and printing the names of the directories and files
for dirpath, dirnames, filename in os.walk(basepath):
    print(f'Found directory: {dirpath}')
    for file in filename:
        path_list.append(os.path.abspath(os.path.join(dirpath, file)))
#%%
def get_cv_email(self, cv_path):
    pagenums = set()
    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)
    infile = open(cv_path, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()
    match = re.search(r'\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b', text)
    email = match.group(0)
    return email

#\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)