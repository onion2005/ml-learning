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
