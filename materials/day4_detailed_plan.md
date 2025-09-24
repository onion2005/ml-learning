# Days 4-6: Advanced MLOps & Enterprise Systems

Building on your foundational MLOps skills from Days 1-3, these final days focus on enterprise-grade systems, advanced deployment patterns, and career-ready portfolio development.

---

## Day 4: Enterprise RAG System & Multi-Model Platform

**Goal**: Build production-ready RAG system with enterprise features
**Time**: 2-3 hours
**Focus**: LLM deployment, vector databases, authentication

### Part 1: RAG System with AWS Bedrock (45 minutes)

#### Step 1: Bedrock Integration Setup (15 minutes)
```python
# notebooks/04_enterprise_rag.ipynb
# Cell 1: AWS Bedrock setup and configuration

import boto3
import json
from typing import List, Dict, Any
import os
from datetime import datetime

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def test_bedrock_access():
    """Test access to Claude via Bedrock"""
    try:
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=json.dumps({
                "messages": [{"role": "user", "content": "Hello, can you respond?"}],
                "max_tokens": 50,
                "anthropic_version": "bedrock-2023-05-31"
            }),
            contentType='application/json'
        )
        result = json.loads(response['body'].read())
        print(f"Bedrock access confirmed: {result['content'][0]['text'][:50]}...")
        return True
    except Exception as e:
        print(f"Bedrock access failed: {e}")
        return False

test_bedrock_access()
```

#### Step 2: Document Processing Pipeline (15 minutes)
```python
# Cell 2: Advanced document processing with chunking strategies

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
import chromadb
from chromadb.config import Settings

class EnterpriseDocumentProcessor:
    def __init__(self):
        self.embeddings = BedrockEmbeddings(
            client=bedrock,
            model_id="amazon.titan-embed-text-v1"
        )
        
        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path="./enterprise_chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
    def create_collection(self, collection_name: str):
        """Create or get document collection"""
        try:
            collection = self.chroma_client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
        except ValueError:
            collection = self.chroma_client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        return collection
        
    def process_documents(self, file_paths: List[str], collection_name: str):
        """Process and store documents with metadata"""
        collection = self.create_collection(collection_name)
        
        # Advanced chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "],
            length_function=len
        )
        
        documents = []
        metadatas = []
        ids = []
        
        for file_path in file_paths:
            # Load document
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
                
            docs = loader.load()
            
            # Split into chunks
            chunks = text_splitter.split_documents(docs)
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{os.path.basename(file_path)}_chunk_{i}"
                
                documents.append(chunk.page_content)
                metadatas.append({
                    "source": os.path.basename(file_path),
                    "chunk_id": i,
                    "file_path": file_path,
                    "timestamp": datetime.now().isoformat(),
                    "chunk_size": len(chunk.page_content)
                })
                ids.append(doc_id)
        
        # Generate embeddings and store
        if documents:
            embeddings = self.embeddings.embed_documents(documents)
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            print(f"Processed {len(documents)} chunks into {collection_name}")
        
        return collection

# Initialize processor
processor = EnterpriseDocumentProcessor()
```

#### Step 3: RAG Query Engine (15 minutes)
```python
# Cell 3: Advanced RAG with reranking and filtering

class EnterpriseRAGEngine:
    def __init__(self, collection):
        self.collection = collection
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        
    def retrieve_context(self, query: str, k: int = 5, filter_dict: Dict = None):
        """Retrieve relevant context with filtering"""
        try:
            # Query vector database
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_dict
            )
            
            contexts = []
            for i in range(len(results['documents'][0])):
                contexts.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
                
            return contexts
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    def generate_answer(self, query: str, contexts: List[Dict], max_tokens: int = 500):
        """Generate answer using Claude via Bedrock"""
        
        # Format context
        context_text = "\n\n".join([
            f"Source: {ctx['metadata']['source']}\nContent: {ctx['content']}"
            for ctx in contexts
        ])
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say so clearly.

Context:
{context_text}

Question: {query}

Please provide a detailed answer based on the context above, and cite which sources you used."""

        try:
            response = self.bedrock.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "anthropic_version": "bedrock-2023-05-31"
                }),
                contentType='application/json'
            )
            
            result = json.loads(response['body'].read())
            return {
                'answer': result['content'][0]['text'],
                'contexts_used': contexts,
                'query': query
            }
            
        except Exception as e:
            print(f"Generation error: {e}")
            return {'answer': 'Error generating response', 'contexts_used': [], 'query': query}
    
    def query(self, question: str, k: int = 5, filter_dict: Dict = None):
        """Complete RAG query pipeline"""
        contexts = self.retrieve_context(question, k, filter_dict)
        if not contexts:
            return {'answer': 'No relevant context found', 'contexts_used': [], 'query': question}
            
        result = self.generate_answer(question, contexts)
        return result

# Test with sample documents (create some test files first)
# collection = processor.process_documents(['path/to/your/documents'], 'knowledge_base')
# rag_engine = EnterpriseRAGEngine(collection)
```

### Part 2: Multi-Tenant Architecture (45 minutes)

#### Step 4: Authentication & Authorization (20 minutes)
```python
# Cell 4: Enterprise authentication system

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional
import hashlib

app = FastAPI(title="Enterprise RAG API", version="2.0.0")
security = HTTPBearer()

# Mock user database (in production, use proper database)
USERS_DB = {
    "admin": {
        "user_id": "admin",
        "email": "admin@company.com", 
        "hashed_password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "tenant_id": "default"
    },
    "user1": {
        "user_id": "user1",
        "email": "user1@company.com",
        "hashed_password": hashlib.sha256("user123".encode()).hexdigest(), 
        "role": "user",
        "tenant_id": "tenant_a"
    }
}

SECRET_KEY = "your-secret-key-here"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = USERS_DB.get(user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
            
        return user
        
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/auth/login")
async def login(username: str, password: str):
    """Authenticate user and return token"""
    user = USERS_DB.get(username)
    if not user or user["hashed_password"] != hashlib.sha256(password.encode()).hexdigest():
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user["user_id"]})
    return {"access_token": access_token, "token_type": "bearer", "user": user}

print("Authentication system configured")
```

#### Step 5: Multi-Tenant RAG Endpoints (25 minutes)
```python
# Cell 5: Tenant-aware RAG API

from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    question: str
    collection_name: Optional[str] = "default"
    max_results: int = 5
    filter_source: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    tenant_id: str
    timestamp: str

class TenantRAGManager:
    def __init__(self):
        self.tenant_collections = {}
        self.processor = EnterpriseDocumentProcessor()
    
    def get_tenant_collection(self, tenant_id: str, collection_name: str = "default"):
        """Get or create tenant-specific collection"""
        key = f"{tenant_id}_{collection_name}"
        
        if key not in self.tenant_collections:
            collection = self.processor.create_collection(key)
            rag_engine = EnterpriseRAGEngine(collection)
            self.tenant_collections[key] = rag_engine
            
        return self.tenant_collections[key]
    
    def query_tenant_data(self, tenant_id: str, query_request: QueryRequest):
        """Query data specific to tenant"""
        rag_engine = self.get_tenant_collection(tenant_id, query_request.collection_name)
        
        # Apply tenant-specific filters
        filter_dict = {"tenant_id": tenant_id} if hasattr(query_request, 'tenant_id') else None
        if query_request.filter_source:
            filter_dict = filter_dict or {}
            filter_dict["source"] = query_request.filter_source
        
        result = rag_engine.query(
            query_request.question, 
            k=query_request.max_results,
            filter_dict=filter_dict
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=[ctx['metadata'] for ctx in result['contexts_used']],
            tenant_id=tenant_id,
            timestamp=datetime.now().isoformat()
        )

# Initialize tenant manager
tenant_manager = TenantRAGManager()

@app.post("/rag/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, current_user: dict = Depends(verify_token)):
    """Query RAG system with tenant isolation"""
    try:
        result = tenant_manager.query_tenant_data(current_user["tenant_id"], request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/upload")
async def upload_documents(
    collection_name: str = "default",
    current_user: dict = Depends(verify_token)
):
    """Upload documents to tenant-specific collection"""
    # In practice, handle file uploads here
    return {
        "message": "Document upload endpoint",
        "tenant_id": current_user["tenant_id"],
        "collection": collection_name
    }

@app.get("/rag/collections")
async def list_collections(current_user: dict = Depends(verify_token)):
    """List collections for current tenant"""
    tenant_id = current_user["tenant_id"]
    collections = [key for key in tenant_manager.tenant_collections.keys() if key.startswith(tenant_id)]
    return {"collections": collections, "tenant_id": tenant_id}

print("Multi-tenant RAG API configured")
```

### Part 3: Monitoring & Observability (30 minutes)

#### Step 6: Advanced Monitoring (30 minutes)
```python
# Cell 6: Comprehensive monitoring system

import logging
from datetime import datetime
import time
from functools import wraps
from typing import Dict, Any
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLOpsMonitor:
    def __init__(self):
        self.metrics = {
            'query_count': 0,
            'total_response_time': 0,
            'error_count': 0,
            'tenant_usage': {},
            'model_performance': {}
        }
        
    def log_query(self, tenant_id: str, query: str, response_time: float, 
                  success: bool, context_count: int):
        """Log query metrics"""
        self.metrics['query_count'] += 1
        self.metrics['total_response_time'] += response_time
        
        if not success:
            self.metrics['error_count'] += 1
            
        # Tenant-specific metrics
        if tenant_id not in self.metrics['tenant_usage']:
            self.metrics['tenant_usage'][tenant_id] = {
                'queries': 0, 'errors': 0, 'avg_response_time': 0
            }
            
        tenant_metrics = self.metrics['tenant_usage'][tenant_id]
        tenant_metrics['queries'] += 1
        if not success:
            tenant_metrics['errors'] += 1
            
        # Update average response time
        tenant_metrics['avg_response_time'] = (
            (tenant_metrics['avg_response_time'] * (tenant_metrics['queries'] - 1) + response_time)
            / tenant_metrics['queries']
        )
        
        # Log structured event
        logger.info(json.dumps({
            'event_type': 'rag_query',
            'tenant_id': tenant_id,
            'query_hash': hashlib.sha256(query.encode()).hexdigest()[:8],
            'response_time_ms': response_time * 1000,
            'success': success,
            'context_count': context_count,
            'timestamp': datetime.now().isoformat()
        }))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        avg_response_time = (
            self.metrics['total_response_time'] / max(self.metrics['query_count'], 1)
        )
        
        return {
            'total_queries': self.metrics['query_count'],
            'error_rate': self.metrics['error_count'] / max(self.metrics['query_count'], 1),
            'avg_response_time_seconds': avg_response_time,
            'tenant_breakdown': self.metrics['tenant_usage'],
            'uptime_hours': time.time() / 3600  # Simplified uptime
        }

# Initialize monitoring
monitor = MLOpsMonitor()

def track_performance(func):
    """Decorator to track function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        error = None
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            response_time = time.time() - start_time
            
            # Extract tenant info from args/kwargs if available
            tenant_id = getattr(kwargs.get('current_user', {}), 'get', lambda x: 'unknown')('tenant_id') or 'unknown'
            query = getattr(kwargs.get('request', {}), 'question', 'unknown')
            context_count = 0  # Would extract from result in real implementation
            
            monitor.log_query(tenant_id, query, response_time, success, context_count)
            
    return wrapper

# Apply monitoring to RAG endpoint
@app.get("/metrics")
async def get_metrics(current_user: dict = Depends(verify_token)):
    """Get system metrics (admin only)"""
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return monitor.get_metrics_summary()

@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check"""
    checks = {
        'bedrock_connectivity': False,
        'database_connectivity': False,
        'vector_db_status': False,
        'memory_usage': 'unknown',
        'disk_usage': 'unknown'
    }
    
    # Test Bedrock connection
    try:
        test_bedrock_access()
        checks['bedrock_connectivity'] = True
    except:
        pass
    
    # Test ChromaDB
    try:
        client = chromadb.PersistentClient(path="./enterprise_chroma_db")
        collections = client.list_collections()
        checks['vector_db_status'] = True
    except:
        pass
    
    overall_status = "healthy" if all(checks.values()) else "degraded"
    
    return {
        'status': overall_status,
        'timestamp': datetime.now().isoformat(),
        'checks': checks,
        'metrics_summary': monitor.get_metrics_summary()
    }

print("Monitoring and observability configured")
```