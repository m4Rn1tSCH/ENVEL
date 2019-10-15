import tkinter

##create a root to hold other widgets
# root (main) window
root = Tk()
# create frame
top = Frame(root)
# pack frame in main window

##widgets need to be packed to show up in the root frame
##orientation of packing
##vertically; (side = 'top' OR side = 'bottom'
##horizontally; (side = 'left' OR side = 'right'
#how to pack top frame is not important
top.pack(side='top')

##place label inside top frame
#placed horizontally; widgets must be placed from left to right
hwtext = Label(top, text='Hello, World! The sine of')
hwtext.pack(side='left')

#create field to hold variable
##"StringVar" needed to tie ordinary variable to content of a widget
#value fields
#StringVar() = string
#DoubleVar() = float

# variable to be attached to widgets
r = StringVar()
r.set('1.2'); # default value
r_entry = Entry(top, width=6, textvariable=r);
r_entry.pack(side='left');

#creating the s variable
# variable to be attached to widgets
s = StringVar()
s_label = Label(top, textvariable=s, width=18)
s_label.pack(side='left')

#defining the button and the function associated with it
##convert the string to a float before calculation
##convert it to a string again before calling before calling s.set

def comp_s():
	global s
	s.set('%g' % math.sin(float(r.get()))) # construct string

compute = Button(top, text=' equals ', command=comp_s)
compute.pack(side='left');
##event loop to keep it updating and running indefinitely
root.mainloop()

