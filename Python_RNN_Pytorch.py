#!/usr/bin/env python
# coding: utf-8




import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

torch.manual_seed(1)

#Hyper parameter
TIME_STEP = 100
INPUT_SIZE = 10
LR = 0.02

#show date
steps = np.linspace(0, np.pi * 2, 100, dtype = np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label = 'target(cos)')
plt.plot(steps, x_np, 'b-', label = 'label(sin)')
plt.legend(loc = 'best')
plt.show()

