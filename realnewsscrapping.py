import csv

import requests
from bs4 import BeautifulSoup
import numpy as np
import urllib.request
import json
class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"
opener = AppURLopener()
news_contents = []
list_links = []
list_titles = []
# # -------------------------------------------FOX NEWS------------------------------------------
# url2='https://www.foxnews.com/category/health/infectious-disease/coronavirus'
# response2 =requests.get(url2)
# if response2.ok:
#    soup2=BeautifulSoup(response2.text,'html5lib')
#    coverpage_news2=soup2.find_all('h4')
#    #find et findal
# url2='https://www.foxnews.com'
# for n in np.arange(0, len(coverpage_news2)):
#        # only news articles (there are also albums and other things)
#        #if "inenglish" not in coverpage_news[n].find('a')['href']:
#         #continue
#
#        # Getting the link of the article
#        link = coverpage_news2[n].find('a')['href']
#        if 'https' not in link :
#            link=url2+link
#        # Reading the content (it is divided in paragraphs)
#        #article = requests.get(link)
#        #if 'video' not in link and 'watch' not in link and not link.startswith("//") :
#         #print(link)
#         # Getting the title
#        title = coverpage_news2[n].find('a').get_text()
#        article = opener.open(link).read()
#         #article_content = article.content
#        soup_article = BeautifulSoup(article, 'html5lib')
#        body = soup_article.find_all('div', class_='article-content')
#        if body:
#          x = body[0].find_all('p')
#          # Unifying the paragraphs
#          list_paragraphs = []
#          for p in np.arange(0, len(x)):
#           paragraph = x[p].get_text()
#           #print(paragraph)
#           list_paragraphs.append(paragraph)
#           final_article = " ".join(list_paragraphs)
#          if final_article :
#           news_contents.append(final_article)
#           list_links.append(link)
#           list_titles.append(title)
#--------------------------------------------ABC NEWS--------------------------
url = 'https://www.abc.net.au/news/story-streams/coronavirus/'
#response2 =requests.get(url)
list_links = []
list_titles = []
news_contents = []
lia = []
with requests.Session() as session:
 headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
 for j in range(1,10):
  response = session.get('https://www.abc.net.au/news-web/api/loader/channelstories?documentId=12015982&prepareParams=%7B%22imagePosition%22:%7B%22mobile%22:%22right%22,%22tablet%22:%22right%22,%22desktop%22:%22right%22%7D%7D&loaderParams=%7B%22pagination%22:%7B%22size%22:5%7D%7D&offset='+str(5*j)+'&size=5&total=50',headers=headers,allow_redirects=False)
  tab=response.json()['collection']
  #print(response.json()['collection'])
 # print(len(tab))
  for o in np.arange(0,len(tab)):
      link='https://www.abc.net.au/'+tab[o]['link']['to']
      list_links.append(link)
#if response2.ok:
 res = session.get(url, headers=headers)
 soup2=BeautifulSoup(res.text,'html5lib')
 lis=soup2.find('div',class_='_3OXQ1 _26IxR _2kxNB i69js')
 h=lis.find_all('a')
 for i in np.arange(0,len(h)):
     link=h[i]["href"]
     if 'https' not in link :
      link='https://www.abc.net.au/'+link
      list_links.append(link)
print(len(list_links))
for p in np.arange(0,len(list_links)):
    res = session.get(list_links[p], headers=headers)
    soup2 = BeautifulSoup(res.text, 'html5lib')
    title=soup2.find('h1').get_text()
    x = soup2.find_all('p')
    list_paragraphs = []
    for j in np.arange(0,len(x)):
     paragraph=x[j].get_text()
     list_paragraphs.append(paragraph)
     final_article = " ".join(list_paragraphs)
    news_contents.append(final_article)
    list_links.append(link)
    list_titles.append(title)
#----------------------------------------------cnet---------------------------------------
url = 'https://www.cnet.com/coronavirus/'
lia = []
for i in np.arange(2,75):
    response2 = requests.get(url+str(i)+'/')
    soup2 = BeautifulSoup(response2.text, 'html5lib')
    lis=soup2.find_all('div',class_='o-card c-premiumCards_card')
    for i in np.arange(0,len(lis)):
     link='https://www.cnet.com'+lis[i].find('a')["href"]
     response2 =requests.get(link)
     soup_article = BeautifulSoup(response2.text, 'html5lib')
     title=soup_article.find('h1')
     x = soup_article.find_all('p')
     list_paragraphs = []
     for j in np.arange(0,len(x)):
      paragraph=x[j].get_text()
      if 'The information contained in this article is for educational' not in paragraph:
       list_paragraphs.append(paragraph)
     final_article = " ".join(list_paragraphs)
     news_contents.append(final_article)
     list_links.append(link)
     title=title.get_text()
     list_titles.append(title)
#-------------------------------------------------BBC---------------------------------------------
#il faut ajouter le numéro de la page par exemple https://www.bbc.com/news/live/explainers-51871385/page/2
url = 'https://www.bbc.com/news/live/explainers-51871385/page/'
l=[]
for i in np.arange(2,35):
    response2 = requests.get(url + str(i))
    soup2 = BeautifulSoup(response2.text, 'html5lib')
    h=soup2.find_all('h3')
    for i in np.arange(0,len(h)):
     try:
      title=h[i].get_text()
      link=h[i].find('a')["href"]
      link='https://www.bbc.com'+link
      response3 = requests.get(link)
      soup3 = BeautifulSoup(response3.text, 'html5lib')
      d=soup3.find_all('div',class_='ssrcss-rgov1k-MainColumn e1sbfw0p0')
      if len(d) == 1:
          l.append(h[i])
          paragraphes=d[0].find_all('p')
          list_paragraphs=[]
          for j in np.arange(0,len(paragraphes)):
              paragraph = paragraphes[j].get_text()
              list_paragraphs.append(paragraph)
          final_article = " ".join(list_paragraphs)
          news_contents.append(final_article)
          list_links.append(link)
          list_titles.append(title)
     except:
      continue
with open('real.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "title", "link", "text"])
    for x in range(0, len(list_links)):
     try:
      writer.writerow([x, list_titles[x], list_links[x], news_contents[x]])
     except:
      continue
