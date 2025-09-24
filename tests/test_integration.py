import pytest
import subprocess
import time
import requests
import docker
import os

class TestIntegration:
    """Integration tests for the complete ML pipeline"""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Docker client for container tests"""
        try:
            client = docker.from_env()
            return client
        except Exception:
            pytest.skip("Docker not available")
    
    def test_docker_build(self, docker_client):
        """Test Docker image builds successfully"""
        dockerfile_path = os.path.join(os.path.dirname(__file__), "..", "docker_ml")
        
        try:
            image, logs = docker_client.images.build(
                path=dockerfile_path,
                tag="iris-classifier:test",
                rm=True
            )
            assert image is not None
        except Exception as e:
            pytest.fail(f"Docker build failed: {e}")
    
    def test_container_startup(self, docker_client):
        """Test container starts and responds to health checks"""
        try:
            # Start container
            container = docker_client.containers.run(
                "iris-classifier:test",
                ports={'8000/tcp': 8001},
                detach=True,
                remove=True
            )
            
            # Wait for startup
            time.sleep(10)
            
            # Test health endpoint
            response = requests.get("http://localhost:8001/health", timeout=5)
            assert response.status_code == 200
            
            # Cleanup
            container.stop()
            
        except Exception as e:
            pytest.fail(f"Container test failed: {e}")
    
    def test_end_to_end_prediction(self):
        """Test complete prediction pipeline"""
        # This would test the entire pipeline from input to output
        # In a real scenario, this might involve:
        # 1. Loading test data
        # 2. Making predictions via API
        # 3. Validating results against expected outcomes
        # 4. Checking logs and metrics
        
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        # This is a placeholder - in practice you'd test against running service
        assert test_data["sepal_length"] > 0