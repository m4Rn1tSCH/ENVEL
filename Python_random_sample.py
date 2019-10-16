#create a random sample
#picking 32 rows randomly (not subsequent ones) from the data and ranking it by date in ascending order (long ago to recent)
#set number of drawn rows/columns, optionally set a weight and a reproducible pseudo-random result

##data = pandas data frame
##GIVE INTEGERS ONLY
sample_size = 32
sample_weights = None
random_state = None

def draw_sample(data, sample_size, sample_weights, random_state):
    #draw the sample and rank it
    random_sample = data.sample(n = sample_size, frac = None, replace = False, weights = sample_weights,
                             random_state = random_state, axis=None)

    #ranking with index (axis = 0)
    ranked_sample = random_sample.rank(axis = 0, method = 'min', numeric_only = None, na_option = 'keep',
                                    ascending = True, pct = False)
    print(ranked_sample.head(5))
    