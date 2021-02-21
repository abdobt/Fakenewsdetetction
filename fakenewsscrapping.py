import csv
import urllib
import numpy as np
from bs4 import BeautifulSoup
import requests
class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"
opener = AppURLopener()
url='https://www.wakingtimes.com/?s=covid+'
list_links = []
list_titles = []
news_contents = []
article = opener.open(url).read()
with requests.Session() as session:
  headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
  res=session.get(url,headers=headers)
  soup2 = BeautifulSoup(res.text, 'html5lib')
  title = soup2.find_all('h2', 'entry-title')
 # for n in np.arange(0, len(title)):
  # print(title[n].find('a')["href"])
  for i in np.arange(2,22):
   response = session.get('https://www.wakingtimes.com/page/'+str(i)+'/?s=covid+',headers=headers)
   soup2 = BeautifulSoup(response.text, 'html5lib')
   title = soup2.find_all('h2', 'entry-title')
   for n in np.arange(0, len(title)):
    link=title[n].find('a')["href"]
    res = session.get(link, headers=headers)
    soup2 = BeautifulSoup(res.text, 'html5lib')
    t=soup2.find('h1')
    paragraphs= soup2.find_all('p')
    list_paragraphs=[]
    for u in np.arange(2,len(paragraphs)):
      paragraph = paragraphs[u].get_text()
      list_paragraphs.append(paragraph)
      final_article = " ".join(list_paragraphs)
    news_contents.append(final_article)
    list_links.append(link)
    list_titles.append(t.get_text())
#fake news scraping ----> done completlty
url='https://www.healthnutnews.com/?s=covid&limit=100'
response2 =requests.get(url)
soup2=BeautifulSoup(response2.text,'html5lib')
lise=soup2.find_all('h2')
for i in np.arange(0,len(lise)):
    link=lise[i].find('a')["href"]
    list_links.append(link)
url='https://www.healthnutnews.com/?s=covid&limit=100&bpaged=100'
response2 =requests.get(url)
soup2=BeautifulSoup(response2.text,'html5lib')
lise=soup2.find_all('h2')
for i in np.arange(0,len(lise)):
    link=lise[i].find('a')["href"]
for j in np.arange(0,len(list_links)):
    response2 = requests.get(list_links[j])
    soup2 = BeautifulSoup(response2.text, 'html5lib')
    try:
     title = soup2.find('h1').get_text()
     a=soup2.find('div',class_='post-content')
     x = a.find_all('p')
     list_paragraphs = []
     for j in np.arange(0, len(x)):
         paragraph = x[j].get_text()
         list_paragraphs.append(paragraph)
         final_article = " ".join(list_paragraphs)
     news_contents.append(final_article)
     list_links.append(link)
     list_titles.append(title)
    except:
     continue
url='https://healthimpactnews.com/wp-admin/admin-ajax.php?action=acs_search_documents&keyword=covid&start=0&size=473&sort_field=_score&sort_order=desc'
response2 =requests.get(url)
soup2=BeautifulSoup(response2.text,'html5lib')
tab=response2.json()['data']['items']
soup2=BeautifulSoup(tab,'html5lib')
a=soup2.find_all('a')
listlinks=[]
for i in np.arange(0,len(a)):
    listlinks.append(a[i]["href"])
listlinks = list(set(listlinks))
for k in np.arange(0,len(listlinks)):
 response=requests.get(listlinks[k])
 link=list_links[k]
 soup2 = BeautifulSoup(response.text, 'html5lib')
 a=soup2.find('h2')
 title=a.get_text()
 div=soup2.find('div',class_='post-content')
 paragraphs=div.find_all('p')
 list_paragraphs = []
 for j in np.arange(3,len(paragraphs)-2):

  if 'See Also' not in paragraphs[j].get_text()  and 'Having problems receiving our newsletters? See:' not in paragraphs[j].get_text()  and 'Comment on this article at' not in paragraphs[j].get_text() and 'Read the full article at' not in paragraphs[j].get_text() :
         paragraph = paragraphs[j].get_text()
         list_paragraphs.append(paragraph)
  else :
      break
 final_article = " ".join(list_paragraphs)
 news_contents.append(final_article)
 list_links.append(link)
 list_titles.append(title)
# greatgameindia done today
l=[]
listlinks=[]
url='https://greatgameindia.com/category/coronavirus-covid19/'
response2 =requests.get(url)
soup2 = BeautifulSoup(response2.text, 'html5lib')
a = soup2.find_all('h3',class_='entry-title')
l=a
for n in np.arange(2,27):
    url = 'https://greatgameindia.com/category/coronavirus-covid19/'+str(n)+'/'
    response2 = requests.get(url)
    soup2 = BeautifulSoup(response2.text, 'html5lib')
    a = soup2.find_all('h3',class_='entry-title')
    l=l+a
for j in np.arange(0,len(l)):
    title=l[j].get_text()
    link=l[j].find('a')["href"]
    listlinks.append(link)
    response2 = requests.get(link)
    soup2 = BeautifulSoup(response2.text, 'html5lib')
    a=soup2.find('div',class_='td-post-content')
    g=a.find_all('p')
    list_paragraphs=[]
    for u in np.arange(0,len(g)-2):
        if 'Enter your email' not in g[u].get_text() and 'Email Address' not in g[u].get_text() and 'Subscribe' not in g[u].get_text():
            paragraph = g[u].get_text()
            list_paragraphs.append(paragraph)
    final_article = " ".join(list_paragraphs)
    news_contents.append(final_article)
    list_links.append(link)
    list_titles.append(title)
#Sauvegarde dans fichier .csv qui sera après le preprocessing ajouté à la base de données
with open('fake.csv', 'w', newline='') as file:
 writer = csv.writer(file)
 writer.writerow(["id", "title", "link", "text"])
 for x in range(0, len(list_links)):
  try:
   writer.writerow([x, list_titles[x], list_links[x], news_contents[x]])
  except:
   continue
