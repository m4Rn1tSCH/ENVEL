#apply Label Encoder from SKLEARN
from sklearn.preprocessing import LabelEncoder
data = 
subset = 
picked_column = data['subset']

def encode_str_to_num:
	#use the LabelEncoder to make shopname numerical
	LE = LabelEncoder()
	data['feat_shopname'] = LE.fit_transform(data['shopname'])
	data.dtypes
	#features feat_shopname + amount ready
