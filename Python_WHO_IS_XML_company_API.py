try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

domainName = 'whoisxmlapi.com';
apiKey = 'at_2Q30rKnR3MAqjchn7J9GPpbAoq9eq'

url = 'https://www.whoisxmlapi.com/whoisserver/WhoisService?'\
    + 'domainName=' + domainName + '&apiKey=' + apiKey + "&outputFormat=JSON"

print(urlopen(url).read().decode('utf8'))