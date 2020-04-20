#Packages
import numpy


json1_file = open('json1')
json1_str = json1_file.read()
json1_data = json.loads(json1_str)[0]
datapoints = numpy.array(json1_data['datapoints'])
avg = datapoints[0:5,0].mean()

