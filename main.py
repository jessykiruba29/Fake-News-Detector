from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
import logging


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class News(BaseModel):
    news:str

@app.post("/news")
async def news_detect(data:News):
    logger.info(f"Received news from user: {data.news}")
   

