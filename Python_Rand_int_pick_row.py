#packages
import torch
import pandas as pd
import numpy as np
test = torch.randn(865,865)
torch.is_tensor(test)

#random data
data = pd.DataFrame(np.random.randn(50,50)).astype('float64')

#randomly pick a line and the 32 following
start_row = np.random.randint(low = 0, high = len(data), size= None, dtype='l')
data[start_row:start_row + 32]

