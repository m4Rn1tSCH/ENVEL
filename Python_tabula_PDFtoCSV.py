import tabula
import os

##SPECIFY INPUT DIRECTORY WITH R IN FRONT
#filename of your PDF/directory where your PDF is stored
link = r"C:\Users\bill-\Desktop\dtc2018_p1.pdf"
new_link = link.replace(os.sep, '/')
input_directory = ''.join(('', new_link,''))
#input_directory = 'C:/Users/bill-/Desktop/MK2019_180807.pdf' #Manual typing

# Read pdf into DataFrame
df = tabula.read_pdf(input_directory, output_format='dataframe', encoding='utf-8',
                     java_options=None, pandas_options=None, multiple_tables=True)

# Rread remote pdf into DataFrame
#df2 = tabula.read_pdf("https://github.com/tabulapdf/tabula-java/raw/master/src/test/resources/technology/tabula/arabic.pdf")

# convert PDF into CSV
tabula.convert_into(input_directory, "dtc2018_p1.csv", output_format="csv")

# convert all PDFs in a directory
#tabula.convert_into_by_batch("C:/Users/bill-/Desktop/", output_format='csv')
