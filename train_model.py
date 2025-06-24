import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
import nltk

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

# model evaluation (training)
X_train_pred=model.predict(X_train)
training_accuracy=accuracy_score(X_train_pred,Y_train)
print(training_accuracy)

# model evaluation (testing)
X_test_pred=model.predict(X_test)
testing_accuracy=accuracy_score(X_test_pred,Y_test)
print(testing_accuracy)

#make a predictor system
input_data="pope passed away"
input_data=stemming(input_data)
input_vector=vectorizer.transform([input_data])

prediction=model.predict(input_vector)
if prediction[0] == 0:
    print("🟢 Real News")
else:
    print("🔴 Fake News")







