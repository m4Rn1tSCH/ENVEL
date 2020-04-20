from bs4 import BeautifulSoup
import urllib.request as ur
import re
import sys

#List of words to avoid in job title
red_flags = ["senior", "intern", "staff", "optimization", "specialist", "Business Development", "Advisor", "Sales"]
#Can also check for required words 
required = ["Financial", "financial", "Analyst", "analyst", "Python", "R-studio", "R", "Investment Banking", "Private Equity"] 

def qualifies(title):
    title = title.lower()
    #Define a function to check if a job title is worth checking out  
    for word in red_flags:
        if word in title: return False
    return True

#test:
qualifies("Financial Analyst")

# Now define the Regex, 
# 1. Should not have the phrase 5-9 Years
p1 = re.compile('[5-9]\s*\+?-?\s*[5-9]?\s*[yY]e?a?[rR][Ss]?')
# 2. Should not have mention of "Citizenship", "Citizens", so on..
p2 = re.compile('[Cc]itizens?(ship)?')

t1 = p1.search("2+ Years of experiencce")
t2 = p1.search("0-1 Year")
print (t1, "\n",t2)

#The first page with search results
url_base = "https://www.linkedin.com/jobs/"
pgno = 0
try:
        response = ur.urlopen(url_base+str(pgno))
        html_doc = response.read()
except:
        print("URL not accessible")
        exit();
soup = BeautifulSoup(html_doc, 'html.parser')
"Ready."
try:
    total_results = soup.find(id="searchCount").get_text()
    last_page = int(int(total_results[total_results.index("of")+2: total_results.index("jobs")].strip()) / 10) * 10
    print(last_page)
except:
    print ("No jobs found")
	
jobs_per_page = 10
goodlinks = []
for pgno in range(0,last_page,jobs_per_page):
    if pgno > 0:
        try:
            response = ur.urlopen(url_base+str(pgno))
            html_doc = response.read()
        except:
            break;
        soup = BeautifulSoup(html_doc, 'html.parser')
    for job in soup.find_all(class_='result'):
        link = job.find(class_="turnstileLink")
        try:
            jt = link.get('title')
        except:
            jt = ""
        try:
            comp = job.find(class_='company').get_text().strip()
        except:
            comp = ""

        if(qualifies(jt.lower())):
            toVisit = "https://www.linkedin.com/jobs/"+link.get('href')
            try:
                html_doc = ur.urlopen(toVisit).read().decode('utf-8')
            except:
                continue;
            m = p1.search(html_doc)
            n = p2.search(html_doc)
            if not m and not n:
                print(jt,",",comp,":",toVisit,"\n")
                goodlinks.append(toVisit)
