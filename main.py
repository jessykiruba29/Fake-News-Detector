from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
import logging
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import dotenv
import os
from fastapi import Request
from starlette.responses import Response


nltk.download("stopwords")
model=joblib.load("logistic_model.pkl")
vectorizer=joblib.load("tfidf_vectorizer.pkl")


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def custom_cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method == "OPTIONS":
        response = Response()
    else:
        response = await call_next(request)

    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
    
    return response

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

    result="The news is True" if prediction[0]==0 else "The news is Fake"
    return {"prediction":result}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    

