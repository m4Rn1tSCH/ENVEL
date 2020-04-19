#Path or filename in the same folder
link = "C:\Users/bill-\Desktop\2016fy detailed Income Statement.pdf" #filename of your PDF/directory where your PDF is stored

filename = link.replace(os.sep, '/')

s = pd.read_csv(filename, header=0, index_col=None, parse_cols=None)
s = pd.read_excel(filepath, sheet_name=0, header=0, index_col=None, parse_cols=None)

#General exploration of data frames
#methods
s.head()
s.info()
s.describe()
s.['col'].mean() #requires dtype = int64
#attributes
s.columns
s.shape
s.value_counts()
s.align()
s = pd.concat([df_1, df_2, df_3], axis = 1) #vertical concatenation 0 = row; 1 = col; None = both 
s_nodup = s.drop_duplicates(keep='first', inplace=False)
#merge and keep identical values
df.where(df1.values==df2.values)
################################################
import pandas as pd

xl = pd.ExcelFile("C:\\Temp\\test.xlsx") # or whatever your filename is

df = xl.parse("Sheet1", header=None)
result = df[0].value_counts()

print(result)
#########################################################