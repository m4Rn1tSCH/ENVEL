#!/usr/bin/env python
#Packages
from tkinter import *
import math
#%%
#core
root = Tk() # root (main) window
top = Frame(root) # create frame
top.pack(side='top') # pack frame in main window

#%%
text = Label(top, text='Hello, World! The sine of')
text.pack(side='left')
r = StringVar() # variable to be attached to r_entry
r.set('1.2') # default value
r_entry = Entry(top, width=5, textvariable=r)
r_entry.pack(side='left')
s = StringVar() # variable to be attached to s_label
#%%

def comp_s():
    global s
    s.set('%g' %math.sin(float(r.get()))) # construct string
    compute = Button(top, text=' equals ', command=comp_s)
    compute.pack(side='left')
    s_label = Label(top, textvariable=s, width=18)
    s_label.pack(side='left')
    root.mainloop()