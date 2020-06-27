#!/usr/bin/env python
# coding: utf-8

# In[251]:


import requests # for making standard html requests
from bs4 import BeautifulSoup # magical tool for parsing html data
import json # for parsing data
import pandas as pd
from pandas import DataFrame as df # premier library for data organization


# In[252]:


column_names = ["IlanNO", "Tarih", "Marka", "Seri"]
df = pd.DataFrame(columns = column_names)


# In[253]:


url ="https://www.arabam.com/ilan/galeriden-satilik-mercedes-benz-e-180-amg/e180-amg-command-cam-tavan-boyasiz-tramersiz-2018-bayi-cikisli/14272378"
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')


# In[254]:


#page.encoding = 'ISO-885901'
#soup = BeautifulSoup(page.text, 'html.parser')


# In[255]:


#print(soup.prettify())


# In[256]:


#ul class="w100 cf mt16"


# In[257]:


arabaOzellikleri = soup.find_all(class_ = 'df df-center w100')
for i in arabaOzellikleri:
  print("arabaOzellikleri:\n", i)
  fiyat= i.find('span')
  print("fiyat:\n", fiyat)  
  print(fiyat.text.strip())
  arabaDic= dict()
  arabaDic['fiyat'] = fiyat.text.strip()


# In[258]:


araba = soup.find_all(class_ = 'w100 cf mt16')
print(araba)


# In[259]:


arabaOzellikleri= araba[0].find_all(['span','a'])

for idx, i  in enumerate(arabaOzellikleri):
  #print("arabaOzellikleri:\n", i)
  if (idx%2==0):
        key = i.text.strip()
  else:
        val = i.text.strip()
        arabaDic[key]=val
    
  print(idx, " : ", i.text.strip())


# In[260]:


#print(arabaOzellikleri.contents)


# In[261]:


#print(arabaOzellikleri[0].attrs)


# In[262]:


arabaDic


# In[263]:


arabaDF= pd.DataFrame(columns = arabaDic.keys())

arabaDF


# In[264]:


arabaDF= arabaDF.append(arabaDic, ignore_index=True)


# In[265]:


arabaDF


# In[266]:


arabaDF.to_csv("test.txt")


# In[ ]:




