##PACKAGES
import pandas as pd
import numpy as np

link = r"C:\Users\bill-\Desktop\TransactionsD.csv"
new_link = link.replace(os.sep, '/')
file = ''.join(('', new_link,''))
##read the data frame
data = pd.read_csv(file, skiprows = 1, index_col = None, names = ['category', 'trans_cat', 'subcat', 'shopname', 'amount'])

##FEATURE ENCODING
#encode the shopname with a numerical value and create the column "feat_shopname"
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
data['feat_shopname'] = LE.fit_transform(data['shopname'])
#No strings allowed only slices and integers!
data.dtypes

data_feat = pd.concat([data['amount'], data['feat_shopname']], axis = 1)
data_feat.dtypes
print(data_feat)

#is not a 2d array but Kmeans takes it
data_array = np.asarray(data_feat)

#DONE
