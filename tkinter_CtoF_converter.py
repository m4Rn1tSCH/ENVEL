# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:01:52 2019

@author: bill-
"""

#version = 1.0
#PURPOSE
#conversion of degrees fahrenheit to degrees celcius and vice versa

#Packages
from tkinter import *
import math
#%%
#CORE
root = Tk() # root (main) window
top = Frame(root) # create frame
top.pack(side='top') # pack frame in main window

#%%
##TEXT
text = Label(top, text = 'Fahrenheit-Celsius Converter')
text.pack(side = 'left')
#fahrenheit input value
f_in = StringVar() # variable to be attached to r_entry
f_in.set('°F input') # default value
f_in_entry = Entry(top, width = 10, textvariable = f_in)
f_in_entry.pack(side = 'left')
#celsius input value
c_in = StringVar() # variable to be attached to r_entry
c_in.set(' °C input') # default value
c_in_entry = Entry(top, width = 10, textvariable = c_in)
c_in_entry.pack(side = 'left')

f = StringVar() # variable to be attached to fahrenheit value
c = StringVar() # variable to be attached to celsius value
#%%
#define functions here and pack them in the frame
#will pack from the left to the right

def deg_fahrenheit():
    global f
    s.set('%g' %float(c_in.get() * 5/9 + 32)) # construct string
#set up the button
    compute = Button(top, text=' to °F ', command=deg_fahrenheit)
    compute.pack(side='left')
    s_label = Label(top, textvariable = c_in, width = 18)
    s_label.pack(side='left')
    root.mainloop()

def deg_celsius():
    global c
    s.set('%g' %float(f_in.get() * 9/5 - 32)) # construct string
#set up the button
    compute = Button(top, text=' to °C ', command=deg_celsius)
    compute.pack(side='left')
    s_label = Label(top, textvariable = f_in, width = 18)
    s_label.pack(side='left')
    root.mainloop()
