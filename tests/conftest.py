import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import joblib
import os

@pytest.fixture
def iris_data():
    """Load iris dataset for testing"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return X, y, iris.target_names

@pytest.fixture
def sample_features():
    """Sample iris features for testing"""
    return {
        'setosa': [5.1, 3.5, 1.4, 0.2],
        'versicolor': [6.0, 2.8, 4.5, 1.3],
        'virginica': [7.2, 3.0, 5.8, 2.3]
    }

@pytest.fixture
def trained_model():
    """Load trained model for testing"""
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    if not model_files:
        pytest.skip("No trained model found")
    
    latest_model = max(model_files, key=lambda f: os.path.getctime(f'models/{f}'))
    model = joblib.load(f'models/{latest_model}')
    return model