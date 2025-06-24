from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
import logging
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk


nltk.download("stopwords")
model=joblib.load("logistic_model.pkl")
vectorizer=joblib.load("tfidf_vectorizer.pkl")


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

port_stem=PorterStemmer()
stop_words=set(stopwords.words('english'))
def stemming(content):
    content=re.sub('[^a-zA-Z]', ' ', content)
    content=content.lower()
    words=content.split()
    stemmed=[port_stem.stem(word) for word in words if word not in stop_words]
    return ' '.join(stemmed)

class News(BaseModel):
    news:str

@app.post("/news")
async def news_detect(data:News):
    logger.info(f"Received news from user: {data.news}")

    processed=stemming(data.news)
    vectorized=vectorizer.transform([processed])
    prediction=model.predict(vectorized)

    result="REAL" if prediction[0]==0 else "FAKE"
    return {"prediction":result}
    

