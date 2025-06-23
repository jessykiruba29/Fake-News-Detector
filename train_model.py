import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
import kagglehub
from kagglehub import KaggleDatasetAdapter
import nltk
nltk.download('stopwords')

file_path = "Fake.csv"
news_list_real=pd.read_csv("./data/True.csv")
news_list_fake=pd.read_csv("./data/Fake.csv")

news_list_real["label"]=0
news_list_fake["label"]=1

news = pd.concat([news_list_real,news_list_fake], ignore_index=True) #concat both into 1 dataframe
 
# REAL => 0
# FAKE => 1


#stemming => to remove prefix suffix, and to only return root word
stem=PorterStemmer()
def stemming(content):
    
    stemmed=re.sub('[^a-zA-Z]',' ',content)
    stemmed=stemmed.lower()
    words=stemmed.split()
    stemmed_words=[stem.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(stemmed_words)

print("Original:", news["text"].iloc[0][:200])
print("Processed:", stemming(news["text"].iloc[0])[:200])

news["text"] = news["text"].apply(stemming)

X = news["text"]
Y = news["label"]

print("✅ X shape:", X.shape)
print("✅ Y shape:", Y.shape)
print("✅ First few samples:")
print(X.head())



