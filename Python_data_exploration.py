import numpy as np
import pandas as pd
##Fill "Non-Available Values " when they are BEFORE no.
data = pd.Series([1, np.nan, 3, np.nan, 5])
data.fillna(method = "bfill")
#OUTPUT
0    1.0
1    3.0
2    3.0
3    5.0
4    5.0
dtype: float64

###############################################

#Complete the code to return the output
import numpy as np
import pandas as pd
##Fill "Non-Available Values " when they are AFTER no.
data = pd.Series([1, 2, np.nan, 4, np.nan])
data.fillna(method = "ffill")

#OUTPUT


0    1.0
1    2.0
2    2.0
3    4.0
4    4.0
dtype: float64

###############################################

Consider the Pandas Series s below:

1       red
2      blue
3       red
dtype: object

#Complete the code to return the output

import pandas as pd
##Count if a string is in a panda series
s.str.contains("d")
#OUTPUT
1     True
2    False
3     True
dtype: bool

###############################################

     gh owner language      repo  stars
0  pandas-dev   python    pandas  17800
1   tidyverse        R     dplyr   2800
2   tidyverse        R   ggplot2   3500
3      has2k1   python  plotnine   1450

#Complete the code to return the output

import pandas as pd
#Regroup a data frame with list comprehension
df[['repo', 'gh owner']]

#OUTPUT
       repo    gh owner
0    pandas  pandas-dev
1     dplyr   tidyverse
2   ggplot2   tidyverse
3  plotnine      has2k1

##############################################

import pandas as pd
#Concatenate 2 panda series to one data frame
s1 = pd.Series(['red', 'blue', 'yellow'], index = [1, 2, 3])
s2 = pd.Series(['purple', 'green', 'orange'], index = [4, 5, 6])
#in square brackets
pd.concat([s1, s2])

###############################################

Consider df below.

   x   y
0  4  16
1  9  25

#Complete the code to return the output
df.apply(lambda col: col * 10)

#OUTPUT


    x    y
0  40  160
1  90  250

###############################################

     gh owner language      repo  stars
0  pandas-dev   python    pandas  17800
1   tidyverse        R     dplyr   2800
2   tidyverse        R   ggplot2   3500
3      has2k1   python  plotnine   1450

#Complete the code to return the output

import pandas as pd
df.stars.apply(lambda x: x / 1000)

#OUTPUT


0    17.80
1     2.80
2     3.50
3     1.45
Name: stars, dtype: float64

###############################################

#Consider the Pandas DataFrame df below.

      Month  Count
zero    Jan     52
one     Apr     29
two     Mar     46
three   Feb      3

#Complete the code to return the output
df.loc['two':]

#OUTPUT


      Month  Count
two     Mar     46
three   Feb      3

###############################################

##Produces a summary with statistics exluding NaNs
df.describe()

###############################################

#Consider following data frame
     gh owner language      repo  stars
0  pandas-dev   python    pandas  17800
1   tidyverse        R     dplyr   2800
2   tidyverse        R   ggplot2   3500
3      has2k1   python  plotnine   1450

#Complete the code to return the output
df.columns

#OUTPUT

Index(['gh owner', 'language', 'repo', 'stars'], dtype='object')

###############################################

from datetime import date

d = date(2019, 5, 1)

d.isoformat()

#OUTPUT

'2019-05-01'

###############################################

#Complete the code to return the output
import datetime as dt
start = dt.datetime(2010, 10, 5, 13, 30, 2)
delta = dt.timedelta(seconds = 2)

print(start + delta)

#OUTPUT

2010-10-05 13:30:04

###############################################

#Consider df_1 and df_2 below.

   repo_id language  stars
0        1   python   1450
1        2        R   2800


   repo_id      repo    gh owner
0        1  plotnine      has2k1
1        3    pandas  pandas-dev

#Complete the code to return the output
pd.merge(df_1, df_2, on = "repo_id", how = 'inner')

#OUTPUT


   repo_id language  stars      repo gh owner
0        1   python   1450  plotnine   has2k1

###############################################

#Complete the code to return the output

x = "Python is a language"
y = x.count("a")

print("There are ", y, "a's.")

###############################################

#Change indices to columns
       eggs
month
a      10
c      15

#Select the code to return the output

cols = ['a', 'b', 'c', 'd']
df = df.reindex(cols)
print(df)

###############################################

#Make everything  upper case

x.upper()

###############################################

#Make everything lower case
x.lower()

###############################################

#Consider the Pandas DataFrame df below.

     gh owner language      repo  stars
0  pandas-dev   python    pandas  17800
1   tidyverse        R     dplyr   2800
2   tidyverse        R   ggplot2   3500
3      has2k1   python  plotnine   1450
#all rows and only gh owner column
df.loc[:,["gh owner"]]

#OUTPUT
     gh owner
0  pandas-dev
1   tidyverse
2   tidyverse
3      has2k1

###############################################

#Consider the Pandas DataFrame df below.

        bmi      country   year
0  21.48678  Afghanistan  Y1980
1  25.22533      Albania  Y1980
2  21.46552  Afghanistan  Y1981
3  25.23981      Albania  Y1981

#Complete the code to return the output

#pivot the table with country as index...
df.pivot(index = "country", columns = "year", values = "bmi")

#OUTPUT
year            Y1980     Y1981
country                        
Afghanistan  21.48678  21.46552
Albania      25.22533  25.23981

###############################################

#Complete the code to return the output

from datetime import date

d = date(2018, 10, 28)
d.year

#OUTPUT
2018

###############################################

#Consider df below.

   x   y
0  4  16
1  9  25

#Complete the code to return the output

df.apply(np.sqrt)

#OUTPUT


     x    y
0  2.0  4.0
1  3.0  5.0

###############################################

#Complete the code to return the output
import numpy as np
import pandas as pd

data = pd.Series([1, 2, np.nan, 4, 5])
data.dropna()

#OUTPUT


0    1.0
1    2.0
3    4.0
4    5.0
dtype: float64

###############################################

#Combine the following DataFrames df1 and df2 :

#df1            #df2

month   eggs     month   eggs
jan      10      mar      12
feb       5      apr      14

#Select the code to return the output

df1.append(df2)


       eggs
month      
jan    10.0
feb     5.0
mar    12.0
apr    14.0


