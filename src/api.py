# src/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np

# --- Configuration ---
MODEL_DIR = "./tfidf_logreg_classifier" # Path within the Docker container
LABEL_MAP = {
    0: 'b (Business)', 
    1: 't (Science/Tech)', 
    2: 'e (Entertainment)', 
    3: 'm (Health)'
}

# Global variables for model and vectorizer
logreg_model = None
tfidf_vectorizer = None

# Pydantic schema for request body
class TextIn(BaseModel):
    text: str

# Pydantic schema for response body
class PredictionOut(BaseModel):
    input_text: str
    predicted_category: str
    label_id: int

@app.on_event("startup")
def load_model():
    """Load the model and vectorizer when the FastAPI application starts."""
    global logreg_model, tfidf_vectorizer
    try:
        logreg_model = joblib.load(os.path.join(MODEL_DIR, 'logreg_model.pkl'))
        tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        print("TF-IDF Model and Vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        
app = FastAPI(title="TF-IDF LogReg News Classifier API")

@app.post("/predict", response_model=PredictionOut)
def predict(request: TextIn):
    """Endpoint for classifying a single news headline."""
    if logreg_model is None or tfidf_vectorizer is None:
        return {"error": "Model not loaded. Check server logs."}
        
    text = [request.text]
    
    # 1. Feature Extraction
    X_tfidf = tfidf_vectorizer.transform(text)
    
    # 2. Run Inference
    predicted_id = logreg_model.predict(X_tfidf)[0]
    
    # 3. Map ID back to category
    predicted_category = LABEL_MAP.get(predicted_id, "Unknown")
    
    return PredictionOut(
        input_text=text[0],
        predicted_category=predicted_category,
        label_id=int(predicted_id)
    )

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": logreg_model is not None}