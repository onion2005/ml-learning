#!/bin/bash

echo "Running ML Pipeline Tests..."

# Install test dependencies
pip install pytest requests docker

# Run different test categories
echo "1. Running unit tests..."
pytest tests/test_model.py -v

echo "2. Running API tests..."
pytest tests/test_api.py -v

echo "3. Running integration tests..."
pytest tests/test_integration.py -v -m "not slow"

echo "Test summary:"
pytest tests/ --tb=line --quiet