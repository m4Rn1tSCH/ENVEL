import PyPDF2
import pandas as pd
import os
##LEAVE THE r in front
link = r"C:\Users\bill-\Desktop\2018 DTC Income Statement FY 2018.pdf" 	#(r for raw link) filename of your PDF/directory where your PDF is stored
new_link = link.replace(os.sep, '/')
PDFfilename = ''.join(('', new_link,''))	#converted link

#PDFfilename = "C:/Users/bill-/Desktop/2017 income statement - fy 2017 final.pdf"
#x = [0, 1, 2, 3, 4, 5]
#slice = x[0:len(x)]


pfr = PyPDF2.PdfFileReader(open(PDFfilename, "rb")) #PdfFileReader object

#extract pg X ZERO-INDEXED= actual - 1 (p2 = 2-1); refer to PDF index not doc index
pages = pfr.getPage(2) 

writer = PyPDF2.PdfFileWriter() #create PdfFileWriter object
#add pages
writer.addPage(pages)

NewPDFfilename = "dtc2018_p3.pdf" #filename of your PDF/directory where you want your new PDF to be
with open(NewPDFfilename, "wb") as outputStream:
	writer.write(outputStream) #write pages to new PDF


	
