#create a random sample


#set number of drawn rows/columns, optionally set a weight and a reproducible pseudo-random result

sample_size = 32
sample_weights = None
random_state = None

def draw_sample:
	#draw the sample and rank it
	random_sample = data.sample(n = sample_size, frac = None, replace = False, weights = sample_weights, random_state = random_state, axis=None)

	#ranking with index (axis = 0)
	ranked_sample = random_sample.rank(axis = 0, method = 'min', numeric_only = None, na_option = 'keep', ascending = True, pct = False)
	print(ranked_sample.head(5))
	df.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)