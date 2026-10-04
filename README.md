# Amazon Review Sentiment API

This project uses Machine Learning and FastAPI to predict whether an Amazon product review is positive or negative.

## Model

The model uses:

- TF-IDF Vectorization
- Logistic Regression

The trained model is saved as `model.pkl`.

## Dataset

The dataset contains 20,000 Amazon reviews.

Features:

- `reviewText` - The text of the customer review

Target:

- `Positive` - 1 for positive reviews and 0 for negative reviews

## Model Performance

The model achieved approximately 88.98% accuracy on the test dataset.

## API Endpoints

### GET /health

Checks whether the API is running and whether the ML model was loaded successfully.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}