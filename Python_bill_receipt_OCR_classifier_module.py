#Packages
import PyPDF2
import pandas as pd
import os
import pyocr
import pyocr.builders
from PIL import Image
import pdf2image

#%%
#list of possible bills
list_bills = list(Eversource, NATIONALGRID, AMAZON)
bills_found = list()
#input link (in APP)

#LEAVE r IN FRONT
#filename of your PDF/directory where your PDF is stored
link = r"C:\Users\bill-\Desktop\OCR_test.pdf"
new_link = link.replace(os.sep, '/')
file_in = ''.join(('', new_link,''))

#conversion to colored image

# SPECIFY INPUT/PDF FOLDER + OUTPUT FOLDER + NO OF PAGES TO BE SCANNED; >>>NOT ZERO-INDEXED<<<
pdf2image.convert_from_path(file_in, dpi=400, output_folder= 'C:/Users/bill-/Desktop/', 
							first_page=0, last_page=5, fmt='png', 
							thread_count=1, userpw=None, use_cropbox=False, strict=False, transparent=False, 
							output_file='4a90245d-9f16-413f-8c8d-23469f4775db')

#OCR process
tools = pyocr.get_available_tools()[0]
text = tools.image_to_string(Image.open(file_in), 
					builder=pyocr.builders.DigitBuilder())

strings = [text]
print(text)

#%%
#search for string in text (Eversource; Nationalgrid etc.)
for names in list_bills:
	
	if text.str.contains(list_bills):

#notification : "is this bill X?"
        bills_found.append(names)
        #menu for confirmation
#IF FAILED
	else:
		pass
#conversion colored image to monochrome mode
#%%
##LEAVE r IN FRONT
#filename of your JPEG/PNG
link = r"C:\Users\bill-\Desktop\4a90245d-9f16-413f-8c8d-23469f4775db-1.png" 
new_link = link.replace(os.sep, '/')
input_directory = ''.join(('', new_link,''))

#opens the JPEG/PNG and converts it to monochrome
img = Image.open(input_directory)
img = img.convert('L')
img.save('image.jpg')
#OCR process
tools = pyocr.get_available_tools()[0]
text = tools.image_to_string(Image.open(input_directory), 
					builder=pyocr.builders.DigitBuilder())

#%%
#search for string
for n in list_bills:
	file_in.str.contains(list_bills)

#notification: "is this bill X?"

#IF FAILED

#ask for manual input
#input for the bill_list to scan
man_bill_list = list()
b = input("Enter your bill or receipt manually :")
man_bill_list.append(b)
print (b , "has been added to the list")
