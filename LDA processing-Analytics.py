
# coding: utf-8

# In[1]:

#Code to read a CSV file and produce the clusters using LDA

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import csv
from nltk.corpus import stopwords

# #Read in the data from the CSV into Python
def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8', errors = 'ignore') for cell in row]

filename = 'C:\Users\Rose Nyameke\OneDrive - North Carolina State University\Classes\Fall 2\Text Mining\Project\sanders.csv'
reader = unicode_csv_reader(open(filename))
header = reader.next()

comments = []

for row in reader:
    comments.append(row[0])
#end: Read in the data from the CSV into Python


#Text Parsing: use NLTK to create the stopwords
stopwords = nltk.corpus.stopwords.words('english')

#adding words to the stop word list manually


#Text Parsing: use NLTK to stem the text
porter = nltk.stem.porter.PorterStemmer()

#Define a function that will tokenize and stem the text using Porter
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [porter.stem(t) for t in filtered_tokens]
    return stems

#Define a function that will tokenize the text
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in comments:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'news_articles', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


#create a pandas DataFrame with the stemmed vocabulary as the index and the tokenized words as the column.
#The benefit of this is it provides an efficient way to look up a stem and return a full token.
#The downside here is that stems to tokens are one to many: the stem 'run' could be associated with 'ran', 'runs', 'running', etc.
#For my purposes this is fine--I'm perfectly happy returning the first token associated with the stem I need to look up.
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print vocab_frame.head()



# In[15]:

#Create the TF-IDF matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#define vectorizer parameters; using n-grams here
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=200000,
                                 min_df=0.10, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(comments) #fit the vectorizer to news articles
print(tfidf_matrix.shape)
indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
features = tfidf_vectorizer.get_feature_names()
top_n = 30
top_features = [features[i] for i in indices[:top_n]]
print top_features


# In[39]:

#k means clustering, outputs number of documents per cluster
from sklearn.cluster import KMeans
terms = tfidf_vectorizer.get_feature_names()
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
clusters = km.labels_.tolist()
films = { 'cluster': clusters}
frame = pd.DataFrame(films, index = [clusters] , columns = ['cluster'])
frame['cluster'].value_counts()


# In[19]:

#Running LDA
import string
from gensim import corpora, models, similarities

#tokenize -- necessary step; we don't use tf-idf for LDA
tokenized_text = [tokenize_and_stem(text) for text in news_articles]

#remove stop words
texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]

from gensim import models
lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary, update_every=5, chunksize=10000, passes=100)

lda.show_topics()
topics_matrix = lda.show_topics(formatted=False, num_words=20)
topics_matrix = np.array(topics_matrix, dtype=object)
topic_words = topics_matrix[:,1]

for i in topic_words:
    print([[str(vocab_frame.loc[word[0]].ix[0,0])] for word in i])
    print()


# In[20]:

docTopic = lda.get_document_topics(corpus,minimum_probability=0)
listDocProb = list(docTopic)

probMatrix = np.zeros(shape=(165,3))
for i,x in enumerate(listDocProb):      #each document i
    for t in x:     #each topic j
        probMatrix[i, t[0]] = t[1]

df = pd.DataFrame(probMatrix)

top_n = 1
topic_d = pd.DataFrame({n: df.T[col].nlargest(top_n).index.tolist()
                  for n, col in enumerate(df.T)}).T
topic_d.columns = ['count']
topic_d['count'].value_counts()



# In[ ]:

#end goal is to output the clusters and then use tableau to create those clusters
