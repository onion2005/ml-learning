#!/bin/bash

echo "Building Docker image..."
docker build -t iris-classifier:latest .

echo "Running container..."
docker run -d \
  --name iris-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  iris-classifier:latest

echo "Waiting for container to start..."
sleep 10

echo "Testing health endpoint..."
curl -f http://localhost:8000/health

echo "Testing prediction endpoint..."
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'

echo "API Documentation available at: http://localhost:8000/docs"
