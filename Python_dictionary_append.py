# Using with ensures that the file is properly closed when you're done
with open('filename.txt', 'rb') as f:
  d = {}
  # Here we use readlines() to split the file into a list where each element is a line
  for line in f.readlines():
    # Now we split the file on `x`, since the part before the x will be
    # the key and the part after the value
    line = line.split('x')
    # Take the line parts and strip out the spaces, assigning them to the variables
    # Once you get a bit more comfortable, this works as well:
    # key, value = [x.strip() for x in line] 
    key = line[0].strip()
    value = line[1].strip()
    # Now we check if the dictionary contains the key; if so, append the new value,
    # and if not, make a new list that contains the current value
    # (For future reference, this is a great place for a defaultdict :)
    if key in d:
      d[key].append(value)
    else:
      d[key] = [value]