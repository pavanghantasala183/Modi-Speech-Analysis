#!/usr/bin/env python
# coding: utf-8

# # This Data set is Obtained from https://www.kaggle.com/abhisheksjmr/speeches-modi. This contains the speeches PM Modi from Aug'2014 to Aug'2020

# In[123]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[16]:


import requests
from bs4 import BeautifulSoup as bs


# In[17]:


import os
os.getcwd()


# In[18]:


os.chdir(r"C:\Users\PAVAN\Desktop\Power BI Udemy")


# In[67]:


speeches = pd.read_csv("PM_Modi_speeches.csv")


# In[20]:


speeches.head()


# In[21]:


speeches.describe()


# In[22]:


speeches.drop(["url","lang","words"], axis = 1, inplace = True)
speeches.head()


# # Word Cloud          Sentiment Analysis           Word Frequency 

# In[29]:


pip install wordcloud


# In[26]:


pip install textblob


# In[30]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
import textblob
import wordcloud
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob


# In[34]:


import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS

def wc(data,bgcolor,title):
    plt.figure(figsize = (100,100))
    wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')


# In[44]:


nltk.download('stopwords')
nltk.download('punkt')


# ### Word Frequency and Word Cloud of Title of the Speech

# In[47]:


from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import re

top_N = 100
#convert list of list into text
#a=''.join(str(r) for v in df_usa['title'] for r in v)

a = speeches_50['title'].str.lower().str.cat(sep=' ')

# removes punctuation,numbers and returns list of words
b = re.sub('[^A-Za-z]+', ' ', a)

#remove all the stopwords from the text
stop_words = list(get_stop_words('en'))         
nltk_words = list(stopwords.words('english'))   
stop_words.extend(nltk_words)

word_tokens = word_tokenize(b)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

# Remove characters which have length less than 2  
without_single_chr = [word for word in filtered_sentence if len(word) > 2]

# Remove numbers
cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]        

# Calculate frequency distribution
word_dist = nltk.FreqDist(cleaned_data_title)
rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['Word', 'Frequency'])

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Word",y="Frequency", data=rslt.head(7))


# In[48]:


wc(cleaned_data_title,'black','Common Words' )


# ### Word Frequency and Word Cloud of Speeches of Modi

# In[49]:


from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import re

top_N = 100
#convert list of list into text
#a=''.join(str(r) for v in df_usa['title'] for r in v)

desc_lower = speeches_50['text'].str.lower().str.cat(sep=' ')

# removes punctuation,numbers and returns list of words
desc_remove_pun = re.sub('[^A-Za-z]+', ' ', desc_lower)

#remove all the stopwords from the text
stop_words = list(get_stop_words('en'))         
nltk_words = list(stopwords.words('english'))   
stop_words.extend(nltk_words)

word_tokens_desc = word_tokenize(desc_remove_pun)
filtered_sentence_desc = [w_desc for w_desc in word_tokens_desc if not w_desc in stop_words]
filtered_sentence_desc = []
for w_desc in word_tokens_desc:
    if w_desc not in stop_words:
        filtered_sentence_desc.append(w_desc)

# Remove characters which have length less than 2  
without_single_chr_desc = [word_desc for word_desc in filtered_sentence_desc if len(word_desc) > 2]

# Remove numbers
cleaned_data_desc = [word_desc for word_desc in without_single_chr_desc if not word_desc.isnumeric()]        

# Calculate frequency distribution
word_dist_desc = nltk.FreqDist(cleaned_data_desc)
rslt_desc = pd.DataFrame(word_dist_desc.most_common(top_N),
                    columns=['Word', 'Frequency'])

#print(rslt_desc)
#plt.style.use('ggplot')
#rslt.plot.bar(rot=0)


plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Word", y="Frequency", data=rslt_desc.head(7))


# In[50]:


wc(cleaned_data_desc,'black','Frequent Words' )


# ### Sentiment Analysis of the Speeches

# In[52]:


from textblob import TextBlob

bloblist_desc = list()

speeches_50_descr_str=speeches_50['text'].astype(str)
for row in speeches_50_descr_str:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    speeches_50_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
 
def f(speeches_50_polarity_desc):
    if speeches_50_polarity_desc['sentiment'] > 0:
        val = "Positive"
    elif speeches_50_polarity_desc['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

speeches_50_polarity_desc['Sentiment_Type'] = speeches_50_polarity_desc.apply(f, axis=1)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=speeches_50_polarity_desc)


# ## PAST 50 SPEECHES

# In[152]:


speeches_50 = speeches.head(50)
speeches_50.head()


# ### Word Count of Economy

# In[159]:


import pandas as pd  

list = speeches_50['text']
  
 
series = pd.Series(list) 
  

count = series.str.count("Economy")

count


# In[160]:


count.aggregate(sum)


# In[161]:


word_count = pd.concat([speeches['date'],count], axis = 1)
word_count.head()


# In[162]:


word_count = word_count.set_index("date")


# In[163]:


word_count.head()


# In[165]:


# Create figure and plot space
fig, ax = plt.subplots(figsize=(10, 10))

# Add x-axis and y-axis
ax.bar(word_count.index.values,
        word_count['text'],
        color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Frequency",
       title="word count of Economy")

# Rotate tick marks on x-axis
plt.setp(ax.get_xticklabels(), rotation=45)

plt.show()


# In[ ]:




