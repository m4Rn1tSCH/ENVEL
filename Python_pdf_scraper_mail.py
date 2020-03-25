# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:39:28 2020

@author: bill-
"""
#installed as pdfminer.six but is loaded as pdfminer
#internal packages seem to be the same
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import re
import os
import csv
import pandas as pd
import shutil
#%%
#set csv_path
basepath = 'C:/Users/bill-/Desktop/Resumes'
basepath_non_hvd = 'C:/Users/bill-/Desktop/Non-Harvard_Resumes'
path_list = []
#error_list = []
#Walking a directory tree and printing the names of the directories and files
for dirpath, dirnames, filename in os.walk(basepath):
    print(f'Found directory: {dirpath}')
    for file in filename:
        path_list.append(os.path.abspath(os.path.join(dirpath, file)))
#%%
#complete the move to folders when exception occurs
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
    match = re.search(r'\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)', text)
    email = match.group(0)
    return email

#\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)
#%%
#works
pure_email_list = []
error_list = []
#source_folder = os.path.relpath(r'C:\Users\bill-\Desktop\Harvard_Resumes\*.csv')
#destination = os.path.relpath(r'C:\Users\bill-\Desktop\Harvard_Resumes_converted')
#destination_failed = os.path.relpath(r'C:\Users\bill-\Desktop\Harvard_Resumes_failed')
'''
try control has to be inside the for loop; while iterating the attempt shall be made
and if an exception occurs, it shall be appended to the list and the loop is to be continued
pattern_1 = r'[\w\.-]+@[\w\.-]+'
pattern_2 = r'\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)'
pattern_3 = r'\d{1,9}+\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)'
'''
for link in path_list:
    try:
        pagenums = set()
        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)
        infile = open(link, 'rb')
        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        infile.close()
        converter.close()
        text = output.getvalue()
        output.close()
        match = re.search(r'\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)', text)
        #match_2 = re.search(r'\d{1,9}+\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)', text)
        email = match.group(0)
        #email_2 = match_2.group(0)
        pure_email_list.append(email)
        #pure_email_list.append(email_2)
        #shutil.move(source_folder, destination)
        #with open('C:/Users/bill-/Desktop/Harvard_mail_list.csv','a') as newFile:
            #newFileWriter=csv.writer(newFile)
#if more than one element; in squared brackets
            #newFileWriter.writerow(email)
#should an exception be hit it will append the bugged PDF path to the error_list
    except:
        error_list.append(link)
        #shutil.move(source_folder,destination_failed)
        pass
print("Following motherfuckers were too fucking dumb to create a properly readable PDF...")
print("-------------------------------------------------------------------------------")
print(error_list)
try:
    pd.DataFrame(pure_email_list).to_csv('C:/Users/bill-/Desktop/WECODE_mail_list.csv')
    pd.DataFrame(error_list).to_csv('C:/Users/bill-/Desktop/WECODE_fail_list.csv')
#if the file is existing catch the error and print it; append the existing file instead creating a new one
except FileExistsError as exc:
    print(exc)
    print("existing file will be appended")
    pd.DataFrame(pure_email_list).to_csv('C:/Users/bill-/Desktop/WECODE_mail_list.csv', mode = 'a', header = False)
    pd.DataFrame(error_list).to_csv('C:/Users/bill-/Desktop/WECODE_fail_list.csv', mode = 'a', header = False)

#%%
pure_email_list = []
error_list = []
#source_folder = os.path.relpath(r'C:\Users\bill-\Desktop\Harvard_Resumes\*.csv')
#destination = os.path.relpath(r'C:\Users\bill-\Desktop\Harvard_Resumes_converted')
#destination_failed = os.path.relpath(r'C:\Users\bill-\Desktop\Harvard_Resumes_failed')
'''
try control has to be inside the for loop; while iterating the attempt shall be made
and if an exception occurs, it shall be appended to the list and the loop is to be continued
pattern_1 = r'[\w\.-]+@[\w\.-]+'
pattern_2 = r'\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)'
pattern_3 = r'\d{1,9}+\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)'
'''
for link in path_list:
    try:
        pagenums = set()
        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)
        infile = open(link, 'rb')
        for page in PDFPage.get_pages(infile, pagenums):
            interpreter.process_page(page)
        infile.close()
        converter.close()
        text = output.getvalue()
        output.close()
        #match = re.search(r'\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)', text)
        match_2 = re.search(r'\d{1,9}+\w+(?:[.-]\w+)*@\w+(?:[.-]\w+)+[.-][a-z_0-9]+(?=[A-Z]|(?!=[.-])\b)', text)
        email = match.group(0)
        #email_2 = match_2.group(0)
        pure_email_list.append(email)
        #pure_email_list.append(email_2)
        #shutil.move(source_folder, destination)
        #with open('C:/Users/bill-/Desktop/Harvard_mail_list.csv','a') as newFile:
            #newFileWriter=csv.writer(newFile)
#if more than one element; in squared brackets
            #newFileWriter.writerow(email)
#should an exception be hit it will append the bugged PDF path to the error_list
    except:
        error_list.append(link)
        #shutil.move(source_folder,destination_failed)
        pass
print("Following motherfuckers were too fucking dumb to create a properly readable PDF...")
print("-------------------------------------------------------------------------------")
print(error_list)
try:
    pd.DataFrame(pure_email_list).to_csv('C:/Users/bill-/Desktop/Non-Harvard_mail_list_2.csv')
    pd.DataFrame(error_list).to_csv('C:/Users/bill-/Desktop/Non-Harvard_fail_list_2.csv')
#if the file is existing catch the error and print it; append the existing file instead creating a new one
except:
    pd.DataFrame(pure_email_list).to_csv('C:/Users/bill-/Desktop/Non-Harvard_mail_list_2.csv', mode = 'a', header = False)
    pd.DataFrame(error_list).to_csv('C:/Users/bill-/Desktop/Non-Harvard_fail_list_2.csv', mode = 'a', header = False)