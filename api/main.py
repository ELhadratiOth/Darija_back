from fastapi import FastAPI, HTTPException
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from .yt_scraper import get_data
import numpy as np
import boto3
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import uuid
from dotenv import load_dotenv
from .main_prepro import tokenize_arab_text

load_dotenv()

required_env_vars = [
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_DEFAULT_REGION',
    'DYNAMODB_TABLE_NAME'
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DynamoDB setup
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)
table = dynamodb.Table(os.getenv('DYNAMODB_TABLE_NAME'))

executor = ThreadPoolExecutor(max_workers=1)
last_save_time = 0

# Initialize models as None
model = None
vectorizer = None

def load_models():
    global model, vectorizer
    
    try:
        # Local model paths - adjust these paths to where your models are stored
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'sentiment_model.joblib')
        vectorizer_path = os.path.join(current_dir, 'models', 'vectorizer.joblib')
        
        # Load models from local files
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Models loaded successfully from local files")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    if not load_models():
        raise Exception("Failed to load models")

def truncate_text(text: str, max_bytes: int = 1024) -> str:
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode('utf-8', errors='ignore')

def save_to_dynamodb(prediction_data):
    global last_save_time

    try:
        current_time = time.time()
        time_since_last_save = current_time - last_save_time
        if time_since_last_save < 1:
            time.sleep(1 - time_since_last_save)
        truncated_text = truncate_text(prediction_data['text'])
            
        item = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'text': truncated_text,
            'sentiment': prediction_data['sentiment'],
            'probability': prediction_data['probability']
        }
        
        table.put_item(Item=item)
        last_save_time = time.time()
    except Exception as e:
        print(f"Error saving to DynamoDB: {str(e)}")

async def async_save_to_dynamodb(prediction_data):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, save_to_dynamodb, prediction_data)

@app.get("/predict/{text}")
async def predict_sentiment(text: str):
    try:
        truncated_text = truncate_text(tokenize_arab_text(text))
        
        text_vectorized = vectorizer.transform([truncated_text])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        probability_percentage = int(round(probabilities[1 if prediction == 1 else 0] * 100))
        
        result = {
            "text": truncated_text,
            "sentiment": "positive" if prediction == 1 else "negative",
            "probability": probability_percentage,
            "truncated": len(text.encode('utf-8')) > 1024
        }
        
        asyncio.create_task(async_save_to_dynamodb(result))
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/analyze-youtube/{video_url}")
async def analyze_youtube_comments(video_url: str, max_results: int = 6):
    try:
        comments_df, thumbnail_url = get_data(video_url, max_results)
        
        analyzed_comments = []
        total_probability = 0
        positive_count = 0
        
        for comment in comments_df["Comment"]:
            text_vectorized = vectorizer.transform([tokenize_arab_text(comment)])
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]
            probability_percentage = int(round(probabilities[1 if prediction == 1 else 0] * 100))
            
            if prediction == 1:
                positive_count += 1
            total_probability += probability_percentage
            
            analyzed_comments.append({
                "text": comment,
                "sentiment": "positive" if prediction == 1 else "negative",
                "probability": probability_percentage
            })
        
        avg_probability = int(round(total_probability / len(analyzed_comments)))
        overall_sentiment = "positive" if positive_count >= len(analyzed_comments)/2 else "negative"
        
        return {
            "url": video_url,
            "thumbnail": thumbnail_url,
            "overall_sentiment": overall_sentiment,
            "average_certainty": avg_probability,
            "comments_analysis": analyzed_comments
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing YouTube URL: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "fin a Batal"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
