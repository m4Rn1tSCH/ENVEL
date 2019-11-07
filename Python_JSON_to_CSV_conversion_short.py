
#load packages
import json
import csv

#dictionary example
employee_data =     '{"employee_details":[{"employee_name": "James", "email": "james@gmail.com", "job_profile": "Sr. Developer"},\
                    {"employee_name": "Smith", "email": "Smith@gmail.com", "job_profile": "Project Lead"},.....]}'

employee_parsed = json.loads(employee_data)

emp_details = employee_parsed['employee_details']

# open a file for writing

data_csv_employees = open('/tmp/EmployData.csv', 'w')

# create the csv writer object

csv_writer = csv.writer(data_csv_employees)

count = 0

for emp in emp_details:

      if count == 0:

             header = emp.keys()

             csv_writer.writerow(header)

             count += 1

      csv_writer.writerow(emp.values())

data_csv_employees.close()
