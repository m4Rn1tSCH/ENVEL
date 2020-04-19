#Path or filename in the same folder
link = "C:\Users\bill-\Desktop\2016fy detailed Income Statement.pdf" #filename of your PDF/directory where your PDF is stored

filename = link.replace(os.sep, '/')

s = pd.read_csv(filename, header=0, index_col=None, parse_cols=None)
s = pd.read_excel(filepath, sheet_name=0, header=0, index_col=None, parse_cols=None)

#General exploration of data frames
df['colname']
#OR
df.colname
	#methods
df.head()
df.tail()
df.info()
df.describe()
df.['col'].mean() #requires dtype = int64
#return a random sample
df.sample()
#drop all records with missing values
df.dropna()
#regroup data
df.groupby(['colname'])
#single brackets to specify column +output is Pandas Series object
#double brackets = output is data frame pandas
df.groupby('rank')[['salary']].mean()
	###attributes
#list the types of the columns
dtypes
#ranke items
df.rank
#list the column names
df.columns
#list the row labels and column names
df.axes
#number of dimensions
df.ndim #[X, Y]
#number of elements
df.size
#return a tuple representing the dimensionality
df.shape
#numpy representation of the data
df.values
#counts all unique values of permutations
df.value_counts()
df.align()
s = pd.concat([df_1, df_2, df_3], axis = 1) #vertical concatenation row = 0; col = 1; None = both 
s_nodup = s.drop_duplicates(keep='first', inplace=False)
#merge and keep identical values
df.where(df1.values==df2.values)

df_1 = pd.read_csv
df_msft = pd.concat([df_1, df_2, df_3, df_4])
df_msft_nodup = df_msft.drop_duplicates(keep='first', inplace=False)
df_msft_nodup.to_csv
df_msft_nodup.to_csv('msft_merged.csv')