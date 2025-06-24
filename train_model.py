import numpy as np
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
import nltk
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
import logging



nltk.download('stopwords')

news=pd.read_csv('./data/news_dataset.csv')
news=news.dropna(subset=["text", "label"])
news["label"]=news["label"].astype(str).str.lower().map({"real": 0, "fake": 1})

port_stem=PorterStemmer()
stop_words=set(stopwords.words('english'))
def stemming(content):
    content=re.sub('[^a-zA-Z]', ' ', content)
    content=content.lower()
    words=content.split()
    stemmed=[port_stem.stem(word) for word in words if word not in stop_words]
    return ' '.join(stemmed)

news["text"]=news["text"].apply(stemming)

X=news["text"]
Y=news["label"]

vectorizer=TfidfVectorizer()
vectorizer.fit(X)
X=vectorizer.transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

# model training
model=LogisticRegression()
model.fit(X_train,Y_train)

joblib.dump(model, "logistic_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved.")













