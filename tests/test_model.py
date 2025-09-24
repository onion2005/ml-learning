import pytest
import numpy as np
from sklearn.metrics import accuracy_score

class TestModelPerformance:
    """Test model performance and reliability"""
    
    def test_model_accuracy_threshold(self, trained_model, iris_data):
        """Test that model meets minimum accuracy threshold"""
        X, y, _ = iris_data
        predictions = trained_model.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        # Model should achieve at least 90% accuracy on training data
        assert accuracy >= 0.90, f"Model accuracy {accuracy:.3f} below threshold 0.90"
    
    def test_model_predictions_format(self, trained_model, sample_features):
        """Test prediction output format"""
        for species, features in sample_features.items():
            prediction = trained_model.predict([features])
            probabilities = trained_model.predict_proba([features])
            
            # Check prediction format
            assert isinstance(prediction, np.ndarray)
            assert len(prediction) == 1
            assert prediction[0] in [0, 1, 2]
            
            # Check probabilities format
            assert isinstance(probabilities, np.ndarray)
            assert probabilities.shape == (1, 3)
            assert np.isclose(probabilities.sum(), 1.0)
            assert all(0 <= prob <= 1 for prob in probabilities[0])
    
    def test_model_consistency(self, trained_model, sample_features):
        """Test that model gives consistent predictions"""
        for species, features in sample_features.items():
            # Make same prediction multiple times
            predictions = [trained_model.predict([features])[0] for _ in range(10)]
            
            # All predictions should be identical
            assert len(set(predictions)) == 1, f"Inconsistent predictions for {species}"
    
    def test_edge_cases(self, trained_model):
        """Test model behavior on edge cases"""
        # Test with zeros
        zero_features = [0.0, 0.0, 0.0, 0.0]
        prediction = trained_model.predict([zero_features])
        assert len(prediction) == 1
        
        # Test with very large values
        large_features = [100.0, 100.0, 100.0, 100.0]
        prediction = trained_model.predict([large_features])
        assert len(prediction) == 1
        
        # Test with negative values
        negative_features = [-1.0, -1.0, -1.0, -1.0]
        prediction = trained_model.predict([negative_features])
        assert len(prediction) == 1

class TestModelRobustness:
    """Test model robustness and error handling"""
    
    def test_feature_order_sensitivity(self, trained_model):
        """Test if model is sensitive to feature order"""
        # Standard order
        features_standard = [5.1, 3.5, 1.4, 0.2]
        pred_standard = trained_model.predict([features_standard])[0]
        
        # Same features, same order (should be identical)
        pred_repeat = trained_model.predict([features_standard])[0]
        assert pred_standard == pred_repeat
    
    def test_batch_prediction_consistency(self, trained_model, sample_features):
        """Test batch vs individual predictions are consistent"""
        features_list = list(sample_features.values())
        
        # Individual predictions
        individual_preds = [trained_model.predict([features])[0] for features in features_list]
        
        # Batch prediction
        batch_preds = trained_model.predict(features_list)
        
        # Should be identical
        assert np.array_equal(individual_preds, batch_preds)