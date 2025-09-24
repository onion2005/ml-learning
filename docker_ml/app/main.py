
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import os
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Iris Classifier API",
    description="ML API for classifying iris flowers",
    version="1.0.0"
)

# Global variables for model
model = None
class_names = ["setosa", "versicolor", "virginica"]

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_id: int
    confidence: float
    probabilities: dict

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model

    # Set MLflow tracking URI from environment or default
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'file:///app/mlruns')
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Try to load from MLflow registry with champion alias
        model_uri = "models:/iris-classifier-best@champion"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded model from MLflow registry")
    except Exception as e:
        logger.warning(f"Could not load from MLflow registry: {e}")

        # Try staging alias as fallback
        try:
            model_uri = "models:/iris-classifier-best/Staging"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Loaded model from MLflow registry (staging)")
        except Exception as e2:
            logger.warning(f"Could not load from MLflow staging: {e2}")

            # Fallback to local model file
            try:
                # Try Docker path first, then local path
                model_paths = [
                    "/app/models/iris_model.pkl",
                    os.path.join(os.path.dirname(__file__), "..", "models", "iris_model.pkl"),
                    os.path.join(os.path.dirname(__file__), "..", "..", "models", "iris_classifier_20250922_113317.pkl")
                ]

                model_loaded = False
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        logger.info(f"Loaded model from local file: {model_path}")
                        model_loaded = True
                        break

                if not model_loaded:
                    logger.error(f"No model found in any of these paths: {model_paths}")
                    raise Exception("No model available")

            except Exception as e3:
                logger.error(f"Failed to load model: {e3}")
                raise e3

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Iris Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """Make prediction for iris classification"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Format response
        response = PredictionResponse(
            predicted_class=class_names[prediction],
            predicted_class_id=int(prediction),
            confidence=float(max(probabilities)),
            probabilities={
                class_names[i]: float(probabilities[i]) 
                for i in range(len(class_names))
            }
        )
        
        logger.info(f"Prediction made: {response.predicted_class}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(features_list: List[IrisFeatures]):
    """Make batch predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare batch input
        input_data = np.array([
            [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]
            for f in features_list
        ])
        
        # Make predictions
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        # Format responses
        responses = []
        for i in range(len(predictions)):
            response = PredictionResponse(
                predicted_class=class_names[predictions[i]],
                predicted_class_id=int(predictions[i]),
                confidence=float(max(probabilities[i])),
                probabilities={
                    class_names[j]: float(probabilities[i][j]) 
                    for j in range(len(class_names))
                }
            )
            responses.append(response)
        
        logger.info(f"Batch prediction made for {len(features_list)} samples")
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
