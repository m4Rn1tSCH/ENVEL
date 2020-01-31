# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:19:34 2020

@author: bill-
"""
'''
Tutorial found on Kaggle.com
accessed on 13th January 2020
https://www.kaggle.com/leonlxy/matplotlib-tutorial-with-exercises
'''

# Before we start, let's import the libraries that will be used in later of this tutorials.
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
#%%
#set random seed so that you could have the exact same results as mine.
np.random.seed(0)

df = pd.DataFrame(data={'a':np.random.randint(0, 100, 30),
                        'b':np.random.randint(0, 100, 30),
                        'c':np.random.randint(0, 100, 30)})
df.head()
#%%
#Let's create a figure and call it fig.
fig = plt.figure()
#This will return an empty figure.

#Let's create a figure with figsize (15, 8) and also call it fig (thus overwriting the reference to the previous fig).
#The 15x8 figsize is arbitrary, but I use it as a standard size to work with for visibility.
#The empty figure can be filld with subplots called axes/ax
#%%
fig = plt.figure(figsize=(15,8))
ax = plt.subplot(1,1,1) # (rows, columns, and location)
                        # this would create a 1x1 grid of subplots
                        # and choose axes #1
#%%
fig = plt.figure(figsize=(15,8))
ax1 = plt.subplot(2,1,1) # this would create a 2x1 grid of subplots
                         # and choose axes #1
ax2 = plt.subplot(2,1,2) # this would create a 2x1 grid of subplots
                         # and choose axes #2
#%%
fig, ax = plt.subplots(2, 1, figsize=(15,8)) # This creates a figure of size 15x8 with
                                             # a 2x1 grid of subplots.
ax[0] # The top axes
ax[1] # The bottom axes
#%%
fig, ax = plt.subplots(2, 2, figsize=(15,8)) # This creates a figure of size 15x8 with
                                             # a 2x1 grid of subplots.

ax[0][0].plot(df.index.values, df['a']) # The top-left axes
ax[0][1].plot(df.index.values, df['b']) # The top-right axes
ax[1][0].plot(df.index.values, df['c']) # The bottom-left axes
ax[1][1].plot(df.index.values, range(len(df))) # The bottom-right axes

#%%
fig, ax = plt.subplots(1,1, figsize=(15,8))

x = df.index.values # The index the dataframe we created up above. Equivalent to [0, 1, ..., 28, 29]
y = df['a'] # Column 'a' from df.

ax.plot(x, y)

#The above plot can be generated without creating the variables
#x and y by passing the values directly to the function.
#%%
fig, ax = plt.subplots(2,1, figsize=(15,8))

ax[0].plot(df.index.values, df['a'])
ax[1].plot(df.index.values, df['b'])
#%%
fig, ax = plt.subplots(1,1, figsize=(15,8))

x = df.index.values # The index the dataframe we created up above. Equivalent to [0, 1, ..., 28, 29]
y1 = df['a'] # Column 'a' from df.
y2 = df['b'] # Column 'a' from df.

ax.plot(x, y1)
ax.plot(x, y2)
#%%
sns.set_style('darkgrid') # setting the plotting style
                          # we only need to call this once,
                          # usually before we start plotting.

fig, ax = plt.subplots(1,1, figsize=(15,8))

ax.plot(df.index.values, df['a'])
ax.plot(df.index.values, df['b'])
#%%
sns.set_style('darkgrid') # setting the plotting style

fig, ax = plt.subplots(1,1, figsize=(15,8))

ax.plot(df.index.values, df['a'], color='red', ls='-.')
#%%
fig, ax = plt.subplots(1,1, figsize=(15,8))

ax.plot(df.index.values, df['a'], label='Line A') # add the label
ax.plot(df.index.values, df['b'], label='Line B') # kwarg to each
ax.plot(df.index.values, df['c'], label='Line C') # function

ax.legend(loc='best') # and now call the ax.legend() function
            # it will read all of the labels from graphical
            # objects under ax
ax.plot(df.index.values, df['b'], color='orange', lw=10)
ax.plot(df.index.values, df['c'], color='yellow', lw=1, marker='o')
#%%
fig, ax = plt.subplots(3,1, figsize=(15,8))

ax[0].plot(df.index.values, df['a'], label='Line A') # Top
ax[1].plot(df.index.values, df['b'], label='Line B') # Middle
ax[2].plot(df.index.values, df['c'], label='Line C') # Bottom

ax[0].legend(loc=4) # This will create a legend for ax[0] in the bottom-right.
ax[1].legend(loc=6) # This will create a legend for ax[1] centre-left.

# Also note that all lines will default to the first color in the default color cycle--blue.

