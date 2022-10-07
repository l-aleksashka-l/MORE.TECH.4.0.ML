import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import os
import requests
from re import sub
from decimal import Decimal
import io
import spacy
import re
from datetime import datetime
import string as S
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import time
from datetime import datetime
import string
import asyncio
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from gensim.models.fasttext import FastText
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer

# в df считываем БД
df = pd.read_csv('ALL.csv')
df = df.text


morph = pymorphy2.MorphAnalyzer()

def lemmatize(text):
    words = text.split() # разбиваем текст на слова
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)

    return res


nltk.download('stopwords')
stop = stopwords.words('russian')

df = df.dropna()
df['text'] = df.apply(lambda x: re.sub(r'[^\w\s]+|[\d]+', r'',x).strip())
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['clean_text_list_lem'] = df['text'].apply(lambda x: lemmatize(x))
print(1)

w2v_model_text = Word2Vec(min_count=1,size=300,alpha=0.03,min_alpha=0.0007, sample=6e-5, sg=1)
tfidfVec = TfidfVectorizer()
w2v_model_text.build_vocab(df['clean_text_list_lem'])
#tfidf = tfidfVec.fit_transform(df['clean_text_list_lem'].apply(lambda x: ' '.join([item for item in x if len(item)>2])))
w2v_model_text.train(df['clean_text_list_lem'], total_examples=w2v_model_text.corpus_count, epochs=30, report_delay=1)
w2v_model_text.init_sims(replace=True)
df['vector_text'] = df['clean_text_list_lem'].apply(lambda x: [w2v_model_text.wv[item]for  item in x])
df['vector_text'] = df['vector_text_tfidf'].apply(lambda x: np.average(x, axis=0))
print(2)

def preparation(text):
    text = pd.Series(text)
    text = text.apply(lambda x: re.sub(r'[^\w\s]+|[\d]+', r'', x).strip())
    text = text.apply(lambda x: x.lower())
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    text = text.apply(lambda x: lemmatize(x))
    return text

def vectorization(preparation):
    vector = preparation.apply(lambda x: [w2v_model_text.wv[item] for item in x  ])
    vector = vector.apply(lambda x: np.average(x, axis=0))
    return vector

#w2v_model_text.

def euclidean(text):
    vector = vectorization(preparation(text))[0]
    print('uik')
    df['euclidean_din'] = df.vector_text.apply(lambda x: np.sqrt(((x - vector)**2).sum()) )
    #TTT['euclidean_din'] = TTT.vector_text.progress_apply(lambda x: np.sqrt(((x - vector)**2).sum()) )

    print(df.nsmallest(7, 'euclidean_din'))
    return df.nsmallest(7, 'euclidean_din')

print(df)
