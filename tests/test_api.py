import pytest
import requests
import json
from fastapi.testclient import TestClient
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'docker_ml', 'app'))

# Set environment variables for testing
os.environ['MLFLOW_TRACKING_URI'] = f"file://{os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'mlruns')}"

try:
    from main import app, load_model
    import asyncio

    # Manually trigger model loading for tests
    asyncio.run(load_model())

    client = TestClient(app)
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
except Exception as e:
    print(f"Warning: Could not initialize app: {e}")
    API_AVAILABLE = False

@pytest.mark.skipif(not API_AVAILABLE, reason="FastAPI app not available")
class TestAPI:
    """Test FastAPI endpoints"""
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_predict_endpoint(self, sample_features):
        """Test prediction endpoint"""
        for species, features in sample_features.items():
            payload = {
                "sepal_length": features[0],
                "sepal_width": features[1],
                "petal_length": features[2],
                "petal_width": features[3]
            }
            
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            
            result = response.json()
            assert "predicted_class" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert result["predicted_class"] in ["setosa", "versicolor", "virginica"]
            assert 0 <= result["confidence"] <= 1
    
    def test_batch_predict_endpoint(self, sample_features):
        """Test batch prediction endpoint"""
        features_list = []
        for features in sample_features.values():
            features_list.append({
                "sepal_length": features[0],
                "sepal_width": features[1],
                "petal_length": features[2],
                "petal_width": features[3]
            })
        
        response = client.post("/predict/batch", json=features_list)
        assert response.status_code == 200
        
        results = response.json()
        assert len(results) == len(features_list)
        
        for result in results:
            assert "predicted_class" in result
            assert "confidence" in result
            assert result["predicted_class"] in ["setosa", "versicolor", "virginica"]
    
    def test_invalid_input(self):
        """Test API error handling"""
        # Missing fields
        payload = {"sepal_length": 5.1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        
        # Invalid data types
        payload = {
            "sepal_length": "invalid",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422