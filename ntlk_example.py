#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:36:40 2020

@author: bill
"""

import requests
import time
import datetime
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,precision_score,recall_score
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
import keras

pd.set_option('display.float_format', lambda x: '%.3f' % x)

url = "https://takealot.zendesk.com/api/v2/search.json"


headers = {
    'authorization': "Basic emVuZGVzay1hZG1pbkBzdXBlcmJhbGlzdC5jb20vdG9rZW46R25VU3EwWkFFcnZRQkgzaWxINWZGQTJpTGxiMmJnUlVTbmJkd05RYQ==",
    'cache-control': "no-cache",
    'postman-token': "a8e453b8-e1af-4b30-ff9f-f38e0cf9ba85"
    }

def clean_text(tweet):
    """
    Utility function to clean tweet text by removing links and special characters
    using simple regex statements.
    """
    tweet = tweet.replace(",", " ")
    tweet = tweet.lower()
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def strip(text):
    text = text.strip()
    return text

def paginate(pg,st,ft,x):
    querystring = {"page":pg,"query":"type:ticket brand:Superbalist created>{0} created<={1}".format(st,ft)}
    response = requests.request("GET", url, headers=headers, params=querystring)
    print(datelist[i], "status  : " , response.status_code)
    json_data = response.json()
    print(json_data['count'])
    for j in json_data['results']:
        try:
            email = j['via']['source']['from']['address']
        except:
            email = "none"
        x.append([j['url'],j['created_at'],j['subject'],j['raw_subject'],j['description'],j['ticket_form_id'],j['via'],j['satisfaction_rating'],j['tags'],email])
    if json_data['next_page']:
        print("page num:", pg)
        pg = pg + 1
        time.sleep(3)
        x = paginate(pg,st,ft,x)
        
    return x

sentence = 

['Hello', 'Shenine', 'How', 'are', 'you']





def remove_stop_words_and_stem(sentence):
    tokens = sentence.split()
    lemmatizer = WordNetLemmatizer()
    sno = nltk.stem.SnowballStemmer('english')
    filtered_sentence = [token for token in tokens if not token in stop_words]
    #pre_final = [sno.stem(token) for token in filtered_sentence]
    pre_final = [lemmatizer.lemmatize(token, pos="v") for token in filtered_sentence]
    final = [token for token in pre_final if token.isalpha()]
   
    return ' '.join(final)



data = []
datelist = pd.date_range(pd.datetime.today() - pd.Timedelta(30,unit='D'), periods=31).tolist()

for i in range(0,len(datelist)):
    datelist[i] = datelist[i].strftime("%Y-%m-%dT00:00:00Z")
    
print(datelist)

for i in range(0,len(datelist)-1):
    stime = datetime.datetime.now()
    xx = paginate(1,datelist[i],datelist[i+1],[])
    data = data + xx
   

    ftime = datetime.datetime.now()
    print(ftime-stime)

model1 = train_cnn(X_train, Y_train, X_test, Y_test)



def train_cnn(X_train, Y_train, X_test, Y_test):
    #num_labels = Y_train.shape[1]
    vocab_size = X_train.shape[1]
    input_length = X_train.shape[0]
    print(X_train.shape)
    print(Y_train.shape)
    batch_size = 100
    
    model = Sequential()
    model.add(Embedding(input_length,
                        64,  # Embedding size
                        input_length=vocab_size))
    model.add(LSTM(100))
    
    #model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
     
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
     
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=10,
                        verbose=1,
                        validation_split=0.2
                       )

def predict_cat(message):
    message = clean_text(message)
    MESSAGE_stemmed = remove_stop_words_and_stem(message)
    
    MESSAGE_tfidf = tfidf.transform(np.array([MESSAGE_stemmed]))   
    predicts = model.predict(MESSAGE_tfidf)
    result = pd.DataFrame(predicts)
    return result

predict_cat("Hello I do not understand how to make a payment via fucking EFT")



