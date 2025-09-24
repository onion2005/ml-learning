
## Day 5: Advanced Deployment & Scaling

**Goal**: Production deployment patterns and infrastructure automation
**Time**: 2-3 hours  
**Focus**: Kubernetes, infrastructure as code, auto-scaling

### Part 1: Kubernetes Deployment (60 minutes)

#### Step 7: Kubernetes Manifests (30 minutes)
```yaml
# Create k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlops-production
  labels:
    environment: production
    team: ml-platform

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-api-config
  namespace: mlops-production
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  MAX_WORKERS: "4"
  BEDROCK_REGION: "us-east-1"

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-api-secrets
  namespace: mlops-production
type: Opaque
stringData:
  JWT_SECRET_KEY: "your-production-secret-key"
  AWS_ACCESS_KEY_ID: "your-access-key"
  AWS_SECRET_ACCESS_KEY: "your-secret-key"

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api-deployment
  namespace: mlops-production
  labels:
    app: rag-api
    version: v2.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
        version: v2.0.0
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-api:v2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: rag-api-config
              key: ENVIRONMENT
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: rag-api-secrets
              key: JWT_SECRET_KEY
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/detailed
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: mlops-production
spec:
  selector:
    app: rag-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: mlops-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Step 8: Infrastructure as Code with Terraform (30 minutes)
```hcl
# terraform/variables.tf
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "mlops-cluster"
}

# terraform/eks.tf
resource "aws_eks_cluster" "mlops_cluster" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = "1.27"

  vpc_config {
    subnet_ids              = aws_subnet.private[*].id
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs    = ["0.0.0.0/0"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_service_policy,
  ]

  tags = {
    Environment = var.environment
    Team        = "ml-platform"
    Purpose     = "mlops"
  }
}

# terraform/node-group.tf
resource "aws_eks_node_group" "mlops_nodes" {
  cluster_name    = aws_eks_cluster.mlops_cluster.name
  node_group_name = "mlops-nodes"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = aws_subnet.private[*].id

  instance_types = ["t3.medium", "t3.large"]
  capacity_type  = "ON_DEMAND"

  scaling_config {
    desired_size = 2
    max_size     = 5
    min_size     = 1
  }

  update_config {
    max_unavailable = 1
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
}

# terraform/rds.tf
resource "aws_rds_instance" "mlops_db" {
  identifier             = "mlops-postgres"
  engine                 = "postgres"
  engine_version        = "14"
  instance_class        = "db.t3.micro"
  allocated_storage     = 20
  storage_encrypted     = true
  
  db_name  = "mlops"
  username = "mlops_user"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.mlops.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  
  tags = {
    Environment = var.environment
    Purpose     = "mlops-metadata"
  }
}

# terraform/outputs.tf
output "cluster_name" {
  value = aws_eks_cluster.mlops_cluster.name
}

output "cluster_endpoint" {
  value = aws_eks_cluster.mlops_cluster.endpoint
}

output "database_endpoint" {
  value = aws_rds_instance.mlops_db.endpoint
}
```

### Part 2: Advanced Monitoring & Alerting (30 minutes)

#### Step 9: Prometheus & Grafana Setup (30 minutes)
```yaml
# monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'rag-api'
      kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
          - mlops-production
      relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        action: keep
        regex: rag-api-service
      - source_labels: [__meta_kubernetes_endpoint_port_name]
        action: keep
        regex: metrics

    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

---
# monitoring/grafana-dashboard.json (embedded in ConfigMap)
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-rag
  namespace: monitoring
data:
  rag-dashboard.json: |
    {
      "dashboard": {
        "title": "RAG API Monitoring",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(http_requests_total{job=\"rag-api\"}[5m])",
                "legendFormat": "{{method}} {{endpoint}}"
              }
            ]
          },
          {
            "title": "Response Time",
            "type": "graph", 
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"rag-api\"}[5m]))",
                "legendFormat": "95th percentile"
              }
            ]
          }
        ]
      }
    }
```

### Part 3: Cost Optimization & Resource Management (30 minutes)

#### Step 10: Auto-scaling and Cost Management (30 minutes)
```python
# Cell 7: Cost optimization monitoring

class CostOptimizationManager:
    def __init__(self):
        self.cost_thresholds = {
            'daily_limit': 50.0,  # $50 per day
            'monthly_limit': 1000.0,  # $1000 per month
            'per_query_limit': 0.10  # $0.10 per query
        }
        
    def estimate_query_cost(self, context_length: int, response_length: int) -> float:
        """Estimate cost for a single query"""
        # Bedrock Claude pricing (approximate)
        input_cost_per_1k = 0.003  # $0.003 per 1K input tokens
        output_cost_per_1k = 0.015  # $0.015 per 1K output tokens
        
        # Rough token estimation (4 chars = 1 token)
        input_tokens = context_length / 4
        output_tokens = response_length / 4
        
        cost = (input_tokens / 1000 * input_cost_per_1k + 
                output_tokens / 1000 * output_cost_per_1k)
        
        return cost
    
    def check_cost_limits(self, current_daily_cost: float, 
                         current_monthly_cost: float) -> Dict[str, Any]:
        """Check if costs are within limits"""
        alerts = []
        
        if current_daily_cost > self.cost_thresholds['daily_limit'] * 0.8:
            alerts.append({
                'type': 'warning',
                'message': f'Daily cost approaching limit: ${current_daily_cost:.2f} / ${self.cost_thresholds["daily_limit"]}'
            })
        
        if current_monthly_cost > self.cost_thresholds['monthly_limit'] * 0.8:
            alerts.append({
                'type': 'warning',
                'message': f'Monthly cost approaching limit: ${current_monthly_cost:.2f} / ${self.cost_thresholds["monthly_limit"]}'
            })
            
        return {
            'within_limits': len(alerts) == 0,
            'alerts': alerts,
            'utilization': {
                'daily': current_daily_cost / self.cost_thresholds['daily_limit'],
                'monthly': current_monthly_cost / self.cost_thresholds['monthly_limit']
            }
        }

cost_optimizer = CostOptimizationManager()

# Integration with FastAPI for cost tracking
@app.middleware("http")
async def cost_tracking_middleware(request, call_next):
    """Track costs for each request"""
    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time
    
    # Estimate cost based on processing time and complexity
    estimated_cost = cost_optimizer.estimate_query_cost(
        context_length=1000,  # Default estimation
        response_length=500
    )
    
    # Add cost header to response
    response.headers["X-Estimated-Cost"] = f"${estimated_cost:.4f}"
    response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
    
    return response

print("Cost optimization system configured")
```