# Day 2: AWS SageMaker & Cloud ML Deployment - Detailed Guide

**Total Time**: 2-3 hours  
**Goal**: Deploy your Day 1 model to AWS SageMaker and understand cloud ML services

**Prerequisites**: Day 1 completed, AWS account configured, trained iris model saved

---

## Part 1: SageMaker Environment Setup (45 minutes)

### Step 1: Upload Your Model to S3 (15 minutes)

```python
# Create new notebook: notebooks/02_sagemaker_deployment.ipynb
# Cell 1: Upload yesterday's model to S3

import boto3
import os
from datetime import datetime
import joblib
import json

# Initialize S3 client
s3 = boto3.client('s3')

# Create S3 bucket for SageMaker (if you didn't yesterday)
bucket_name = f"ml-learning-sagemaker-{datetime.now().strftime('%Y%m%d')}"
region = 'ap-southeast-2'

try:
    # Create bucket
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={'LocationConstraint': region}
    )
    print(f"✅ Created S3 bucket: {bucket_name}")
except Exception as e:
    if 'BucketAlreadyExists' in str(e):
        print(f"📁 Bucket {bucket_name} already exists")
    else:
        print(f"⚠️ Error creating bucket: {e}")

# Find your saved model from yesterday
model_files = [f for f in os.listdir('../models') if f.endswith('.pkl')]
if not model_files:
    print("❌ No model files found. Please complete Day 1 first.")
else:
    latest_model = max(model_files, key=lambda f: os.path.getctime(f'../models/{f}'))
    model_path = f'../models/{latest_model}'
    
    # Upload model to S3
    s3_model_key = f'models/{latest_model}'
    s3.upload_file(model_path, bucket_name, s3_model_key)
    
    # Upload metadata too
    metadata_file = latest_model.replace('.pkl', '_metadata.json')
    metadata_path = f'../models/{metadata_file}'
    if os.path.exists(metadata_path):
        s3_metadata_key = f'models/{metadata_file}'
        s3.upload_file(metadata_path, bucket_name, s3_metadata_key)
    
    print(f"✅ Uploaded model to s3://{bucket_name}/{s3_model_key}")
    
    # Save these for later use
    with open('../config.json', 'w') as f:
        json.dump({
            'bucket_name': bucket_name,
            'model_s3_key': s3_model_key,
            'region': region
        }, f)
```

### Step 2: Create SageMaker Execution Role (15 minutes)

```python
# Cell 2: Set up SageMaker IAM role

import boto3
import json

iam = boto3.client('iam')
role_name = 'SageMakerExecutionRole-MLLearning'

# Define the trust policy for SageMaker
trust_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}

# Define permissions policy
permissions_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                f"arn:aws:s3:::{bucket_name}",
                f"arn:aws:s3:::{bucket_name}/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "cloudwatch:PutMetricData"
            ],
            "Resource": "*"
        }
    ]
}

try:
    # Create the role
    role_response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description='SageMaker execution role for ML learning'
    )
    role_arn = role_response['Role']['Arn']
    print(f"✅ Created role: {role_arn}")
    
    # Create and attach policy
    policy_response = iam.create_policy(
        PolicyName='SageMakerMLLearningPolicy',
        PolicyDocument=json.dumps(permissions_policy)
    )
    
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn=policy_response['Policy']['Arn']
    )
    print("✅ Attached permissions policy")
    
except Exception as e:
    if 'EntityAlreadyExists' in str(e):
        # Role already exists, get its ARN
        role_response = iam.get_role(RoleName=role_name)
        role_arn = role_response['Role']['Arn']
        print(f"📁 Using existing role: {role_arn}")
    else:
        print(f"⚠️ Error creating role: {e}")
        # Fallback to using SageMaker's default service role
        role_arn = f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/service-role/AmazonSageMaker-ExecutionRole-*"

# Save role ARN for later
config = json.load(open('../config.json'))
config['role_arn'] = role_arn
with open('../config.json', 'w') as f:
    json.dump(config, f)
    
print(f"🎯 SageMaker setup complete!")
print(f"   Bucket: {bucket_name}")
print(f"   Role: {role_arn}")
```

### Step 3: Create SageMaker Session (15 minutes)

```python
# Cell 3: Initialize SageMaker

import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel

# Load configuration
config = json.load(open('../config.json'))

# Create SageMaker session
sagemaker_session = sagemaker.Session(default_bucket=config['bucket_name'])

print(f"📋 SageMaker Session Information:")
print(f"   Default bucket: {sagemaker_session.default_bucket()}")
print(f"   Region: {sagemaker_session.boto_region_name}")
print(f"   Role ARN: {config['role_arn']}")

# Test SageMaker access
try:
    # List any existing models
    sm_client = boto3.client('sagemaker')
    models = sm_client.list_models(MaxResults=5)
    print(f"✅ SageMaker access confirmed")
    print(f"   Found {len(models['Models'])} existing models in account")
except Exception as e:
    print(f"⚠️ SageMaker access issue: {e}")

print("\n🚀 Ready to deploy your first model to SageMaker!")
```

**Checkpoint 1**: ✅ SageMaker environment configured with S3 bucket, IAM role, and session

---

## Part 2: Model Deployment to SageMaker (45 minutes)

### Step 4: Create SageMaker-Compatible Model Script (20 minutes)

```python
# Cell 4: Create model inference script

# SageMaker requires specific entry point scripts
# Create the script that SageMaker will use to load and run your model

inference_script = '''
import joblib
import json
import numpy as np
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    """Load the model from the model_dir. This is called once per worker."""
    import os
    model_path = os.path.join(model_dir, 'iris_model.pkl')
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data for inference."""
    if request_content_type == 'application/json':
        # Parse JSON input
        input_data = json.loads(request_body)
        
        # Handle different input formats
        if isinstance(input_data, list):
            # Direct list of features: [5.1, 3.5, 1.4, 0.2]
            return np.array([input_data])
        elif isinstance(input_data, dict):
            if 'instances' in input_data:
                # Batch format: {"instances": [[5.1, 3.5, 1.4, 0.2], [...]]}
                return np.array(input_data['instances'])
            else:
                # Named features: {"sepal_length": 5.1, "sepal_width": 3.5, ...}
                features = [
                    input_data.get('sepal_length', 0),
                    input_data.get('sepal_width', 0),
                    input_data.get('petal_length', 0),
                    input_data.get('petal_width', 0)
                ]
                return np.array([features])
    
    elif request_content_type == 'text/csv':
        # Parse CSV input
        df = pd.read_csv(StringIO(request_body), header=None)
        return df.values
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction using the loaded model."""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    # Return both prediction and confidence
    results = []
    class_names = ['setosa', 'versicolor', 'virginica']
    
    for i, pred in enumerate(predictions):
        results.append({
            'predicted_class': class_names[pred],
            'predicted_class_id': int(pred),
            'confidence': float(max(probabilities[i])),
            'probabilities': {
                class_names[j]: float(probabilities[i][j]) 
                for j in range(len(class_names))
            }
        })
    
    return results

def output_fn(prediction, content_type):
    """Format the prediction output."""
    if content_type == 'application/json':
        return json.dumps(prediction), content_type
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''

# Save the inference script
os.makedirs('../sagemaker_code', exist_ok=True)
with open('../sagemaker_code/inference.py', 'w') as f:
    f.write(inference_script)

print("✅ Created SageMaker inference script")
print("📄 Script includes:")
print("   - model_fn: Loads your trained model")
print("   - input_fn: Parses incoming requests")
print("   - predict_fn: Makes predictions")
print("   - output_fn: Formats responses")
```

### Step 5: Package and Deploy Model (15 minutes)

```python
# Cell 5: Deploy model to SageMaker endpoint

import tarfile
import shutil

# Load configuration
config = json.load(open('../config.json'))

# Copy your trained model to deployment directory
deployment_dir = '../sagemaker_model'
os.makedirs(deployment_dir, exist_ok=True)

# Copy the model file
model_files = [f for f in os.listdir('../models') if f.endswith('.pkl')]
latest_model = max(model_files, key=lambda f: os.path.getctime(f'../models/{f}'))
shutil.copy(f'../models/{latest_model}', f'{deployment_dir}/iris_model.pkl')

print(f"✅ Prepared model for deployment: {latest_model}")

# Create model.tar.gz for SageMaker
with tarfile.open(f'{deployment_dir}/model.tar.gz', 'w:gz') as tar:
    tar.add(f'{deployment_dir}/iris_model.pkl', arcname='iris_model.pkl')

# Upload model package to S3
model_artifacts_key = 'sagemaker-models/iris-model.tar.gz'
s3.upload_file(
    f'{deployment_dir}/model.tar.gz',
    config['bucket_name'],
    model_artifacts_key
)

model_artifacts_uri = f"s3://{config['bucket_name']}/{model_artifacts_key}"
print(f"✅ Model artifacts uploaded to: {model_artifacts_uri}")

# Create SageMaker model
from sagemaker.sklearn.model import SKLearnModel

sklearn_model = SKLearnModel(
    model_data=model_artifacts_uri,
    role=config['role_arn'],
    entry_point='inference.py',
    source_dir='../sagemaker_code',
    framework_version='1.2-1',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

print("🚀 Ready to deploy endpoint...")
```

### Step 6: Deploy Real-time Endpoint (10 minutes)

```python
# Cell 6: Create SageMaker endpoint

print("🚀 Deploying model to SageMaker endpoint...")
print("⏳ This typically takes 3-5 minutes...")

endpoint_name = f'iris-classifier-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

try:
    # Deploy the model to an endpoint
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',  # Cost-effective for learning
        endpoint_name=endpoint_name
    )
    
    print(f"✅ Endpoint deployed successfully!")
    print(f"   Endpoint name: {endpoint_name}")
    print(f"   Instance type: ml.t2.medium")
    print(f"   Instance count: 1")
    
    # Save endpoint info
    config['endpoint_name'] = endpoint_name
    with open('../config.json', 'w') as f:
        json.dump(config, f)
    
except Exception as e:
    print(f"❌ Deployment failed: {e}")
    print("💡 This might be due to:")
    print("   - AWS service limits")
    print("   - Insufficient permissions")
    print("   - Region-specific issues")
    print("   - We'll continue with batch processing instead")
```

**Checkpoint 2**: ✅ Model deployed to SageMaker real-time endpoint

---

## Part 3: Testing and Monitoring (30 minutes)

### Step 7: Test Your Deployed Model (15 minutes)

```python
# Cell 7: Test the deployed endpoint

# Test data - different iris flower measurements
test_cases = [
    {
        "name": "Typical Setosa",
        "features": [5.1, 3.5, 1.4, 0.2],
        "expected": "setosa"
    },
    {
        "name": "Typical Versicolor", 
        "features": [6.0, 2.8, 4.5, 1.3],
        "expected": "versicolor"
    },
    {
        "name": "Typical Virginica",
        "features": [7.2, 3.0, 5.8, 2.3],
        "expected": "virginica"
    },
    {
        "name": "Edge case",
        "features": [5.8, 2.7, 4.1, 1.0],
        "expected": "versicolor or virginica"
    }
]

print("🧪 Testing deployed model with various inputs...")
print("=" * 60)

if 'endpoint_name' in config:
    for test_case in test_cases:
        try:
            # Make prediction
            result = predictor.predict(test_case["features"])
            
            print(f"\n🌸 Test: {test_case['name']}")
            print(f"   Input: {test_case['features']}")
            print(f"   Expected: {test_case['expected']}")
            print(f"   Predicted: {result[0]['predicted_class']}")
            print(f"   Confidence: {result[0]['confidence']:.2%}")
            
            # Show all probabilities
            print("   Probabilities:")
            for species, prob in result[0]['probabilities'].items():
                print(f"     {species}: {prob:.2%}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
else:
    print("⚠️ No endpoint available. Testing with local model instead...")
    # Fallback to local testing
    local_model = joblib.load(f'../models/{latest_model}')
    class_names = ['setosa', 'versicolor', 'virginica']
    
    for test_case in test_cases:
        features = np.array([test_case['features']])
        prediction = local_model.predict(features)[0]
        probabilities = local_model.predict_proba(features)[0]
        
        print(f"\n🌸 Test: {test_case['name']}")
        print(f"   Input: {test_case['features']}")
        print(f"   Expected: {test_case['expected']}")
        print(f"   Predicted: {class_names[prediction]}")
        print(f"   Confidence: {max(probabilities):.2%}")

print("\n✅ Model testing completed!")
```

### Step 8: Monitor Endpoint Performance (15 minutes)

```python
# Cell 8: Set up monitoring and cost tracking

import time

# Check endpoint status and metrics
if 'endpoint_name' in config:
    try:
        # Get endpoint details
        endpoint_info = sm_client.describe_endpoint(EndpointName=config['endpoint_name'])
        
        print(f"📊 Endpoint Status: {endpoint_info['EndpointStatus']}")
        print(f"📅 Created: {endpoint_info['CreationTime']}")
        print(f"🔧 Instance type: {endpoint_info['ProductionVariants'][0]['InstanceType']}")
        
        # Get CloudWatch metrics
        cloudwatch = boto3.client('cloudwatch')
        
        # Check invocations in the last hour
        metrics = cloudwatch.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName='Invocations',
            Dimensions=[
                {'Name': 'EndpointName', 'Value': config['endpoint_name']},
                {'Name': 'VariantName', 'Value': 'AllTraffic'}
            ],
            StartTime=datetime.now() - pd.Timedelta(hours=1),
            EndTime=datetime.now(),
            Period=300,  # 5 minute intervals
            Statistics=['Sum']
        )
        
        total_invocations = sum([point['Sum'] for point in metrics['Datapoints']])
        print(f"📈 Total invocations in last hour: {total_invocations}")
        
        # Estimate costs
        cost_per_hour = 0.065  # ml.t2.medium pricing (approximate)
        hours_running = 1  # Since we just deployed
        estimated_cost = cost_per_hour * hours_running
        
        print(f"💰 Estimated cost so far: ${estimated_cost:.4f}")
        print(f"💰 Cost per hour: ${cost_per_hour}")
        print(f"⚠️ Remember to delete endpoint when done learning!")
        
    except Exception as e:
        print(f"⚠️ Monitoring setup incomplete: {e}")

# Set up basic alerting
print("\n🔔 Setting up cost alert...")

try:
    # Create CloudWatch alarm for endpoint invocations
    cloudwatch.put_metric_alarm(
        AlarmName=f'SageMaker-HighInvocations-{config["endpoint_name"]}',
        ComparisonOperator='GreaterThanThreshold',
        EvaluationPeriods=1,
        MetricName='Invocations',
        Namespace='AWS/SageMaker',
        Period=300,
        Statistic='Sum',
        Threshold=1000.0,
        ActionsEnabled=False,  # Just monitoring, no actions
        AlarmDescription='Alert when endpoint gets high traffic',
        Dimensions=[
            {'Name': 'EndpointName', 'Value': config['endpoint_name']},
            {'Name': 'VariantName', 'Value': 'AllTraffic'}
        ],
        Unit='Count'
    )
    print("✅ CloudWatch alarm created for high invocations")
except Exception as e:
    print(f"⚠️ Could not create alarm: {e}")

print("\n📋 Day 2 Monitoring Summary:")
print(f"   ✅ Model deployed to: {config.get('endpoint_name', 'local')}")
print(f"   ✅ Real-time predictions working")
print(f"   ✅ Basic monitoring configured")
print(f"   ✅ Cost tracking enabled")
```

**Checkpoint 3**: ✅ Model tested and monitoring configured

---

## Part 4: Cleanup and Cost Management (15 minutes)

### Step 9: Understanding SageMaker Costs

```python
# Cell 9: Cost analysis and optimization

print("💰 SageMaker Cost Breakdown:")
print("=" * 40)

# Endpoint costs
print("🔹 Real-time Endpoint (ml.t2.medium):")
print("   Cost: ~$0.065/hour ($1.56/day)")
print("   Use case: Real-time predictions")
print("   When to use: Production applications")

# Batch transform costs  
print("\n🔹 Batch Transform (alternative):")
print("   Cost: Pay per compute time only")
print("   Use case: Batch processing")
print("   When to use: Large datasets, scheduled jobs")

# Training costs
print("\n🔹 Training Jobs:")
print("   Cost: Pay per training time")
print("   Use case: Model development")
print("   Note: We used local training (free)")

print("\n💡 Cost Optimization Tips:")
print("   1. Delete endpoints when not in use")
print("   2. Use batch transform for large batches")
print("   3. Choose appropriate instance types")
print("   4. Monitor usage with CloudWatch")
print("   5. Set up billing alerts")

# Calculate costs for different scenarios
scenarios = {
    "Learning (1 day)": {"hours": 24, "instance": "ml.t2.medium", "cost_per_hour": 0.065},
    "Development (1 week)": {"hours": 168, "instance": "ml.t2.medium", "cost_per_hour": 0.065},
    "Production (1 month)": {"hours": 720, "instance": "ml.m5.large", "cost_per_hour": 0.192}
}

print("\n📊 Cost Scenarios:")
for scenario, details in scenarios.items():
    total_cost = details["hours"] * details["cost_per_hour"]
    print(f"   {scenario}: ${total_cost:.2f} ({details['instance']})")
```

### Step 10: Cleanup Resources (Optional)

```python
# Cell 10: Clean up resources to avoid charges

print("🧹 Resource Cleanup Options:")
print("=" * 40)

cleanup_choice = input("Do you want to keep the endpoint running? (y/N): ").lower().strip()

if cleanup_choice != 'y':
    if 'endpoint_name' in config:
        try:
            print("🗑️ Deleting SageMaker endpoint...")
            predictor.delete_endpoint()
            print("✅ Endpoint deleted successfully")
            
            # Update config
            del config['endpoint_name']
            with open('../config.json', 'w') as f:
                json.dump(config, f)
                
        except Exception as e:
            print(f"⚠️ Cleanup issue: {e}")
            print("💡 You can manually delete in AWS Console > SageMaker > Endpoints")
    
    print("\n💰 After cleanup, you're only paying for:")
    print("   - S3 storage (~$0.023/GB/month)")
    print("   - CloudWatch logs (minimal)")
    
else:
    print("\n⚠️ Endpoint will continue running and incurring charges")
    print("💡 Remember to delete it later to avoid costs")
    print(f"   Endpoint name: {config.get('endpoint_name')}")

print("\n🎯 Day 2 Complete! Summary:")
print("   ✅ Learned SageMaker deployment")
print("   ✅ Created real-time ML endpoint") 
print("   ✅ Tested cloud-based predictions")
print("   ✅ Set up monitoring and cost tracking")
print("   ✅ Understanding of AWS ML costs")
```

---

## Day 2 Summary & Next Steps

### What You Accomplished

```python
# Cell 11: Day 2 achievements summary

achievements = [
    "✅ Deployed your local model to AWS SageMaker",
    "✅ Created production-ready real-time endpoint",
    "✅ Built SageMaker inference scripts",
    "✅ Tested cloud predictions with various inputs",
    "✅ Set up CloudWatch monitoring and alerting",
    "✅ Learned AWS ML cost management",
    "✅ Configured S3 for model artifacts",
    "✅ Created proper IAM roles for ML workflows"
]

for achievement in achievements:
    print(achievement)

print(f"\n🎯 Key Concepts Mastered:")
print("   - SageMaker model deployment lifecycle")
print("   - Real-time vs batch inference trade-offs")
print("   - MLOps infrastructure on AWS")
print("   - Production monitoring and cost optimization")

print(f"\n🚀 Tomorrow's Preview (Day 3):")
print("   - MLflow for experiment tracking")
print("   - Model versioning and registry")
print("   - Automated CI/CD for ML models")
print("   - Docker containerization for ML")

print(f"\n💼 Career Relevance:")
print("Today you demonstrated core MLOps skills that companies need:")
print("   - Cloud ML deployment experience")
print("   - Production monitoring setup")
print("   - Cost management awareness")
print("   - AWS infrastructure knowledge")
```

### Skills Gained Today

**Technical Skills:**
- SageMaker model deployment
- Real-time endpoint creation and management
- CloudWatch monitoring for ML systems
- S3 integration for model artifacts
- IAM configuration for ML workflows

**Business Skills:**
- AWS ML cost analysis and optimization
- Production readiness considerations
- Monitoring and alerting strategy
- Resource cleanup and cost control

### Tomorrow's Preparation

```bash
# Quick prep for Day 3
echo "Day 3 preparation:" > day3_prep.md
echo "- [ ] SageMaker deployment working ✅" >> day3_prep.md
echo "- [ ] AWS costs understood ✅" >> day3_prep.md
echo "- [ ] Install Docker (if not already installed)" >> day3_prep.md
echo "- [ ] Read about MLflow (10 minutes)" >> day3_prep.md
```

**Estimated AWS Costs for Day 2**: $1-3 (if endpoint runs for few hours)
**Time Investment**: 2-3 hours
**Value**: Hands-on cloud ML deployment experience

You now have practical experience with AWS ML services - a key requirement for MLOps roles!
