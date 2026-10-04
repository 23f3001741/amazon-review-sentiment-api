from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# Create FastAPI application
app = FastAPI(
    title="Amazon Review Sentiment API",
    description="API for predicting whether an Amazon review is positive or negative.",
    version="1.0"
)


# Load trained model
try:
    model = joblib.load("model.pkl")
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    load_error = str(e)


# Request format for /predict
class ReviewRequest(BaseModel):
    reviewText: str


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: ReviewRequest):

    if model is None:
        return {
            "error": "Model could not be loaded"
        }

    prediction = model.predict([request.reviewText])[0]

    if prediction == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return {
        "prediction": int(prediction),
        "sentiment": sentiment
    }