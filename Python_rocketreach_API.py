import requests
import json

host = 'https://api.rocketreach.co'
url = '{host}/v1/api/lookupProfile'.format(host=host)
headers = {
'API-KEY': '229582k0095d682a8217db65d3f4129a3003b65',
}
payload = {
'name': 'Mark Zuckerberg',
'current_employer': 'Facebook',
}
# Packages the request, send the request and catch the response: r
r = requests.get(url, headers=headers, params=payload)

with open('r', 'w') as outfile:  
    json.dump(r, outfile)

