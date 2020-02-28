# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:38:15 2020

@author: bill-
"""
'''
This script reads all files of a folder and converts them to a PNG picture;
eventually it reads all pictures with optical character recognition and searches for keywords
'''
#required package
import pdf2image
import os
import pyocr
import pyocr.builders
import pandas as pd
from PIL import Image
import os

#LEAVE r IN FRONT
#filename of your PDF/directory where your PDF is stored
#link = r"C:\Users\bill-\Desktop\Harvard Resumes"
#new_link = link.replace(os.sep, '/')
#input_pdf = ''.join(('', new_link,''))
#%%
#import files and append all directory paths to a list
import os
basepath = 'C:/Users/bill-/Desktop/Harvard_Resumes'
path_list = []
#Walking a directory tree and printing the names of the directories and files
for dirpath, dirnames, filename in os.walk(basepath):
    print(f'Found directory: {dirpath}')
    for file in filename:
        if os.path.isfile(file):
            print("file found and appended")
        path_list.append(os.path.abspath(os.path.join(dirpath, file)))

#%%
for path in path_list:
    pdf2image.convert_from_path(path, dpi = 500, output_folder = 'C:/Users/bill-/Desktop/Harvard_Resumes_converted',
                        first_page = 0, last_page = 2, fmt = 'png',
                        thread_count = 1, userpw = None, use_cropbox = False, strict = False, transparent = False,
                        output_file = '4a90245d-9f16-413f-8c8d-23469f4775db')

#%%
#LEAVE r INF FRONT
#filename of your PDF/directory where your PNG is stored;regularly indexed
#link = r"C:\Users\bill-\Desktop\4a90245d-9f16-413f-8c8d-23469f4775db-1.png"

#new_link = link.replace(os.sep, '/')
#input_directory = ''.join(('', new_link,''))

#tools = pyocr.get_available_tools()[0]


#text = tools.image_to_string(Image.open(input_directory),
#                            builder=pyocr.builders.DigitBuilder())

#print(text)

#INACTIVE
#OCR_list =[]
#OCR_list_full = OCR_list.append(text)
#OCR_DF = pd.DataFrame(OCR_list_full)
#OCR_DF.to_csv("C:/Users/bill-/Desktop/OCR_text.csv")


