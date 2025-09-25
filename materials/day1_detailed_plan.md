# Day 1: Environment Setup & ML Basics - Detailed Guide

**Total Time**: 2-3 hours  
**Goal**: Set up complete ML development environment and train your first model

---

## Part 1: AWS Environment Setup (45 minutes)

### Step 1: AWS Account & IAM Setup (15 minutes)

```bash
# 1. Create or use existing AWS account for learning
# Recommendation: Use your existing account but create dedicated user

# 2. Create dedicated IAM user for ML learning
aws iam create-user --user-name ml-learning-user

# 3. Create and attach policy for ML services
cat > ml-learning-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:*",
                "s3:*",
                "iam:GetRole",
                "iam:PassRole",
                "ecr:*",
                "logs:*",
                "cloudwatch:*",
                "bedrock:*",
                "lambda:*"
            ],
            "Resource": "*"
        }
    ]
}
EOF

aws iam create-policy --policy-name MLLearningPolicy --policy-document file://ml-learning-policy.json

# 4. Attach policy to user
aws iam attach-user-policy --user-name ml-learning-user --policy-arn arn:aws:iam::YOUR_ACCOUNT:policy/MLLearningPolicy

# 5. Create access keys
aws iam create-access-key --user-name ml-learning-user
```

### Step 2: S3 Bucket for ML Assets (10 minutes)

```bash
# Create dedicated S3 bucket for ML learning
BUCKET_NAME="ml-learning-$(date +%Y%m%d)-$(whoami)"
aws s3 mb s3://$BUCKET_NAME --region ap-southeast-2

# Create folder structure
aws s3api put-object --bucket $BUCKET_NAME --key data/
aws s3api put-object --bucket $BUCKET_NAME --key models/
aws s3api put-object --bucket $BUCKET_NAME --key notebooks/
aws s3api put-object --bucket $BUCKET_NAME --key artifacts/

echo "Created bucket: $BUCKET_NAME"
echo "Save this bucket name: $BUCKET_NAME" > bucket_name.txt
```

### Step 3: Local AWS Configuration (10 minutes)

```bash
# Configure AWS CLI for ML user
aws configure --profile ml-learning
# Enter the access key and secret from Step 1
# Region: ap-southeast-2
# Output format: json

# Test configuration
aws sts get-caller-identity --profile ml-learning

# Set environment variable for easy use
echo "export AWS_PROFILE=ml-learning" >> ~/.bashrc
echo "export ML_BUCKET_NAME=$BUCKET_NAME" >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Cost Monitoring Setup (10 minutes)

```bash
# Set up billing alert (since you care about cost optimization)
aws budgets create-budget --account-id YOUR_ACCOUNT_ID --budget '{
    "BudgetName": "ML-Learning-Budget",
    "BudgetLimit": {
        "Amount": "50",
        "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
}'

# Create cost allocation tags
aws ce create-cost-category-definition \
    --name "MLLearning" \
    --rules '[{
        "Value": "Learning",
        "Rule": {
            "Tags": {
                "Key": "Project",
                "Values": ["MLLearning"]
            }
        }
    }]'
```

**Checkpoint 1**: ✅ AWS environment configured with dedicated user, S3 bucket, and cost monitoring

---

## Part 2: Local Development Environment (30 minutes)

### Step 5: Python Environment Setup (15 minutes)

```bash
# Create project directory
mkdir -p ~/ml-learning/day1
cd ~/ml-learning/day1

# Create virtual environment (using your preferred method)
python3 -m venv ml-env
source ml-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core ML dependencies
cat > requirements.txt << 'EOF'
# Core ML libraries
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.4
matplotlib==3.8.2
seaborn==0.13.0

# AWS SDK
boto3==1.34.0
sagemaker==2.199.0

# Jupyter for experimentation
jupyter==1.0.0
ipykernel==6.26.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
EOF

pip install -r requirements.txt

# Set up Jupyter kernel
python -m ipykernel install --user --name ml-env --display-name "ML Learning Environment"
```

### Step 6: Environment Configuration (10 minutes)

```bash
# Create .env file for configuration
cat > .env << EOF
AWS_PROFILE=ml-learning
ML_BUCKET_NAME=ml-learning-20250922-cloudshell-user
AWS_DEFAULT_REGION=ap-southeast-2
OPENAI_API_KEY=your_key_here_if_you_have_one
EOF

# Create basic project structure
mkdir -p {data,notebooks,src,models,experiments}
touch src/__init__.py

# Create git repository
git init
cat > .gitignore << 'EOF'
.env
ml-env/
__pycache__/
*.pyc
.DS_Store
.jupyter/
data/raw/
models/*.pkl
*.log
EOF

git add .
git commit -m "Initial ML learning project setup"
```

### Step 5: Test Environment (5 minutes)

```python
# Create test_environment.py
cat > test_environment.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify ML environment setup"""

import sys
import importlib

def test_imports():
    """Test that all required packages can be imported"""
    required_packages = [
        'sklearn', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'boto3', 'sagemaker', 'jupyter'
    ]
    
    print("Testing package imports...")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    return True

def test_aws_connection():
    """Test AWS connection"""
    try:
        import boto3
        session = boto3.Session()
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS connection successful")
        print(f"   Account: {identity['Account']}")
        print(f"   User: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"❌ AWS connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing ML Learning Environment")
    print("=" * 40)
    
    if test_imports() and test_aws_connection():
        print("\n🎉 Environment setup successful!")
        print("You're ready to start ML development!")
    else:
        print("\n❌ Environment setup needs attention")
        sys.exit(1)
EOF

# Run the test
python test_environment.py
```

**Checkpoint 2**: ✅ Local development environment configured with Python, ML libraries, and AWS connectivity

---

## Part 3: ML Fundamentals Crash Course (45 minutes)

### Step 7: Understanding ML Workflow (15 minutes)

```python
# Create notebooks/01_ml_basics.ipynb
jupyter notebook
```

**In Jupyter notebook, create and run these cells:**

```python
# Cell 1: Imports and setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

print("📚 ML Basics - Understanding the Workflow")
print("="*50)
```

```python
# Cell 2: Load and explore data
# Using Iris dataset as it's simple and well-understood
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("🌸 Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Features: {list(iris.feature_names)}")
print(f"Target classes: {list(iris.target_names)}")
print("\nFirst 5 rows:")
print(df.head())

# Quick visualization
plt.figure(figsize=(10, 6))
sns.pairplot(df, hue='species_name', diag_kind='hist')
plt.suptitle('Iris Dataset - Feature Relationships', y=1.02)
plt.show()
```

```python
# Cell 3: Understand the ML workflow
print("🔄 Machine Learning Workflow:")
print("1. Data Collection ✅ (We have iris dataset)")
print("2. Data Exploration ✅ (We examined shape, features, distribution)")
print("3. Data Preparation (Next - split into train/test)")
print("4. Model Selection (Next - choose algorithm)")
print("5. Model Training (Next - fit model to data)")
print("6. Model Evaluation (Next - test performance)")
print("7. Model Deployment (Later - put model in production)")
print("8. Model Monitoring (Later - track performance over time)")

# This workflow applies to ALL ML projects, from simple classification to complex LLMs
```

### Step 8: Build Your First Model (20 minutes)

```python
# Cell 4: Data preparation
print("🛠️ Step 3: Data Preparation")

# Separate features (X) and target (y)
X = df[iris.feature_names]  # Features: sepal length, sepal width, etc.
y = df['species']           # Target: species class (0, 1, 2)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data into training and testing sets
# This is crucial - we never let the model see test data during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training split: {X_train.shape[0]/(X_train.shape[0]+X_test.shape[0]):.1%}")
```

```python
# Cell 5: Model selection and training
print("🤖 Step 4 & 5: Model Selection and Training")

# Choose Random Forest - good general-purpose algorithm
# In production, you'd try multiple algorithms and compare
model = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    random_state=42,     # For reproducible results
    max_depth=3          # Prevent overfitting
)

print(f"Selected algorithm: {type(model).__name__}")
print("Key parameters:")
print(f"- Trees: {model.n_estimators}")
print(f"- Max depth: {model.max_depth}")

# Train the model (this is where the magic happens)
print("\n🎯 Training model...")
model.fit(X_train, y_train)
print("✅ Training completed!")

# The model has now learned patterns from the training data
```

```python
# Cell 6: Model evaluation
print("📊 Step 6: Model Evaluation")

# Make predictions on test data (data the model has never seen)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# Detailed classification report
print("\n📈 Detailed Performance:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix - shows exactly what the model got right/wrong
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Model Predictions vs Actual')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

```python
# Cell 7: Understanding what the model learned
print("🧠 Understanding Model Insights")

# Feature importance - which features matter most for predictions
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("📊 Feature Importance (what the model learned):")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance - What Matters Most for Species Classification')
plt.xlabel('Importance Score')
plt.show()

print("\n🎯 Key Takeaways:")
print("- Petal length is most important for classifying iris species")
print("- Petal width is second most important")
print("- This makes biological sense - petals vary more between species")
print("- Model achieved >95% accuracy - very good for this dataset")
```

### Step 9: Save and Version Your Model (10 minutes)

```python
# Cell 8: Model persistence and versioning
import joblib
import json
from datetime import datetime
import os

print("💾 Step 7: Model Persistence (Saving for Later Use)")

# Create models directory
os.makedirs('../models', exist_ok=True)

# Save the trained model
model_filename = f"../models/iris_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(model, model_filename)
print(f"✅ Model saved to: {model_filename}")

# Save model metadata (crucial for production systems)
metadata = {
    "model_type": "RandomForestClassifier",
    "algorithm": "Random Forest",
    "dataset": "Iris",
    "features": list(iris.feature_names),
    "target_classes": list(iris.target_names),
    "accuracy": float(accuracy),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "parameters": {
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "random_state": model.random_state
    },
    "created_date": datetime.now().isoformat(),
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__
}

metadata_filename = model_filename.replace('.pkl', '_metadata.json')
with open(metadata_filename, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata saved to: {metadata_filename}")
print("\n📝 This metadata is crucial for:")
print("- Model versioning and tracking")
print("- Reproducing results")
print("- Understanding model performance")
print("- Compliance and auditing")
print("- Debugging production issues")
```

```python
# Cell 9: Test model loading (simulating production use)
print("🔄 Testing Model Loading (Production Simulation)")

# Load the saved model
loaded_model = joblib.load(model_filename)

# Load metadata
with open(metadata_filename, 'r') as f:
    loaded_metadata = json.load(f)

print(f"✅ Loaded model created on: {loaded_metadata['created_date']}")
print(f"✅ Model accuracy: {loaded_metadata['accuracy']:.2%}")

# Test prediction with new data (simulating real-world use)
# Create a new iris flower measurement
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Sample measurements

prediction = loaded_model.predict(new_flower)
prediction_proba = loaded_model.predict_proba(new_flower)

predicted_species = iris.target_names[prediction[0]]
confidence = max(prediction_proba[0])

print(f"\n🌸 New flower prediction:")
print(f"   Measurements: {new_flower[0]}")
print(f"   Predicted species: {predicted_species}")
print(f"   Confidence: {confidence:.2%}")

print(f"\n🎉 Congratulations! You've completed your first ML project!")
print("You now understand:")
print("- The complete ML workflow")
print("- How to train and evaluate models")
print("- How to save and load models for production")
print("- How to make predictions on new data")
```

**Checkpoint 3**: ✅ First ML model trained, evaluated, and saved with proper versioning

---

## Part 4: Connect to AWS (Optional - 15 minutes)

### Step 10: Upload Model to S3 (Optional)

```python
# Cell 10: Upload to AWS S3 (if time permits)
import boto3
from dotenv import load_dotenv

load_dotenv()

print("☁️ Uploading Model to AWS S3")

try:
    s3 = boto3.client('s3')
    bucket_name = os.getenv('ML_BUCKET_NAME')
    
    # Upload model file
    model_s3_key = f"models/day1/{os.path.basename(model_filename)}"
    s3.upload_file(model_filename, bucket_name, model_s3_key)
    
    # Upload metadata
    metadata_s3_key = f"models/day1/{os.path.basename(metadata_filename)}"
    s3.upload_file(metadata_filename, bucket_name, metadata_s3_key)
    
    print(f"✅ Model uploaded to s3://{bucket_name}/{model_s3_key}")
    print(f"✅ Metadata uploaded to s3://{bucket_name}/{metadata_s3_key}")
    
    # This is the foundation for model deployment and versioning in production
    
except Exception as e:
    print(f"⚠️ AWS upload failed (that's okay for now): {e}")
    print("You can set this up later as we progress")
```

---

## Day 1 Summary & Reflection

### What You Accomplished Today

```python
# Cell 11: Day 1 Summary
print("🎯 Day 1 Achievements Summary")
print("="*50)

achievements = [
    "✅ Set up complete ML development environment",
    "✅ Configured AWS services for ML learning",
    "✅ Understanding the complete ML workflow",
    "✅ Trained your first machine learning model",
    "✅ Achieved >95% accuracy on classification task",
    "✅ Learned model evaluation and interpretation",
    "✅ Implemented model versioning and persistence",
    "✅ Tested model loading and prediction",
    "✅ Connected local environment to AWS (optional)"
]

for achievement in achievements:
    print(achievement)

print(f"\n📊 Today's Results:")
print(f"   Model Accuracy: {accuracy:.2%}")
print(f"   Dataset: {len(df)} samples, {len(iris.feature_names)} features")
print(f"   Algorithm: Random Forest with {model.n_estimators} trees")
print(f"   Time Invested: ~2-3 hours")

print(f"\n🚀 Tomorrow's Preview (Day 2):")
print("   - AWS SageMaker introduction")
print("   - Deploy your model to the cloud")
print("   - Set up model endpoints for real-time predictions")
print("   - Learn about managed ML services")

print(f"\n💡 Key Insight:")
print("You've just completed the foundation of every ML system.")
print("Whether it's a simple classifier or a complex LLM like ChatGPT,")
print("the workflow is the same: data → train → evaluate → deploy → monitor")
```

### Quick Knowledge Check

Answer these questions to confirm your understanding:

1. **What are the 7 steps of the ML workflow?**
2. **Why do we split data into training and testing sets?**
3. **What does model accuracy of 95% mean?**
4. **What is feature importance and why is it useful?**
5. **Why is model metadata important for production systems?**

### Next Steps Preparation

```bash
# Create Day 2 preparation
echo "🔮 Day 2 Preparation Checklist:" > day2_prep.md
echo "- [ ] AWS account configured ✅" >> day2_prep.md
echo "- [ ] First model trained ✅" >> day2_prep.md
echo "- [ ] Development environment ready ✅" >> day2_prep.md
echo "- [ ] Understand ML workflow ✅" >> day2_prep.md
echo "- [ ] Review SageMaker documentation (15 mins)" >> day2_prep.md
echo "- [ ] Ensure AWS credits/budget for tomorrow" >> day2_prep.md

cat day2_prep.md
```

---

## Troubleshooting Common Issues

### If AWS Setup Fails
```bash
# Check AWS configuration
aws configure list --profile ml-learning
aws sts get-caller-identity --profile ml-learning

# Verify permissions
aws iam get-user --user-name ml-learning-user
```

### If Python Environment Issues
```bash
# Reset virtual environment
deactivate
rm -rf ml-env
python3 -m venv ml-env
source ml-env/bin/activate
pip install -r requirements.txt
```

### If Jupyter Doesn't Start
```bash
# Reset Jupyter kernel
jupyter kernelspec uninstall ml-env
python -m ipykernel install --user --name ml-env --display-name "ML Learning Environment"
jupyter notebook --no-browser --port=8888
```

**Total Time Invested**: 2-3 hours  
**AWS Costs**: ~$0-5 (mostly free tier)  
**Value Created**: Foundation for $200K+ MLOps career  

You're now ready for Day 2! 🚀