# Production Deployment Guide

Enterprise deployment patterns and infrastructure requirements for Temporal Bandwidth Optimizer.

## Overview

This guide covers deploying TBO in production environments with enterprise-grade reliability, monitoring, and scaling.

## Architecture Requirements

### Infrastructure Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   TBO Service   │────│   LLM APIs      │
│                 │    │   Instances     │    │   (Claude/GPT)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   Redis Cache   │              │
         │              │   Cluster       │              │
         │              └─────────────────┘              │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Logging       │    │   Metrics       │
│   (Prometheus)  │    │   (ELK Stack)   │    │   (DataDog)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Latency Budget Allocation

| Component | Target Latency | Budget Allocation |
|-----------|----------------|-------------------|
| Network | < 50ms | 10% |
| Load Balancer | < 10ms | 2% |
| TBO Processing | < 150ms | 30% |
| LLM API Call | < 300ms | 58% |

**Total Target: < 500ms**

## Deployment Options

### 1. Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY examples/ ./examples/
COPY setup.py .

# Install TBO
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import tboptimizer; print('OK')" || exit 1

# Run application
CMD ["python", "-m", "tboptimizer.server"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  tbo-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 512M

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - tbo-app

volumes:
  redis_data:
```

### 2. Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: tbo-system
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: tbo-config
  namespace: tbo-system
data:
  config.yaml: |
    optimization:
      level: "balanced"
      target_latency: 0.5
      cache_size: 10000
    
    monitoring:
      enabled: true
      health_check_interval: 60
      metrics_export_interval: 300
    
    redis:
      url: "redis://redis-service:6379"
      pool_size: 20
```

#### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tbo-deployment
  namespace: tbo-system
spec:
  replicas: 5
  selector:
    matchLabels:
      app: tbo
  template:
    metadata:
      labels:
        app: tbo
    spec:
      containers:
      - name: tbo
        image: tbo:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: anthropic-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
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
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
      volumes:
      - name: config-volume
        configMap:
          name: tbo-config
```

#### Service and Ingress

```yaml
apiVersion: v1
kind: Service
metadata:
  name: tbo-service
  namespace: tbo-system
spec:
  selector:
    app: tbo
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tbo-ingress
  namespace: tbo-system
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - tbo.yourdomain.com
    secretName: tbo-tls
  rules:
  - host: tbo.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tbo-service
            port:
              number: 80
```

### 3. Cloud Deployment

#### AWS ECS/Fargate

```yaml
# task-definition.json
{
  "family": "tbo-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "tbo-container",
      "image": "your-ecr-repo/tbo:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "REDIS_URL",
          "value": "redis://elasticache-endpoint:6379"
        }
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:anthropic-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/tbo",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### GCP Cloud Run

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: tbo-service
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "2"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 100
      containers:
      - image: gcr.io/PROJECT/tbo:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: anthropic-secret
              key: api-key
        resources:
          limits:
            cpu: "1"
            memory: "2Gi"
```

## Configuration Management

### Production Configuration

```python
# config/production.py
import os
from tboptimizer import ProductionConfig

PRODUCTION_CONFIG = ProductionConfig(
    optimization_level="balanced",
    target_latency=0.5,
    cache_size=50000,
    enable_monitoring=True,
    max_concurrent_requests=500,
    rate_limit_per_minute=10000,
    circuit_breaker_threshold=2.0,
    health_check_interval=30,
    metrics_export_interval=60
)

# Redis configuration
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0)),
    "password": os.getenv("REDIS_PASSWORD"),
    "socket_keepalive": True,
    "socket_keepalive_options": {},
    "connection_pool_kwargs": {
        "max_connections": 100,
        "retry_on_timeout": True
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "json",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "/var/log/tbo/app.log",
            "maxBytes": 10485760,
            "backupCount": 5
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

### Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key

# Application
TBO_LOG_LEVEL=INFO
TBO_OPTIMIZATION_LEVEL=balanced
TBO_TARGET_LATENCY=0.5
TBO_CACHE_SIZE=50000

# Infrastructure
REDIS_URL=redis://redis-cluster:6379
DATABASE_URL=postgresql://user:pass@db:5432/tbo

# Monitoring
PROMETHEUS_GATEWAY=http://prometheus-pushgateway:9091
DATADOG_API_KEY=your_datadog_key
SENTRY_DSN=your_sentry_dsn

# Security
JWT_SECRET_KEY=your_jwt_secret
API_RATE_LIMIT=1000
```

## Monitoring and Observability

### Prometheus Metrics

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Request metrics
REQUEST_COUNT = Counter(
    'tbo_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'tbo_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# Optimization metrics
CACHE_HIT_RATE = Gauge(
    'tbo_cache_hit_rate',
    'Cache hit rate'
)

BANDWIDTH_EFFICIENCY = Gauge(
    'tbo_bandwidth_efficiency',
    'Collaborative bandwidth efficiency'
)

CIRCUIT_BREAKER_STATE = Gauge(
    'tbo_circuit_breaker_active',
    'Circuit breaker active state'
)

# Business metrics
COLLABORATION_SESSIONS = Gauge(
    'tbo_active_sessions',
    'Number of active collaboration sessions'
)

def export_metrics(client):
    """Export TBO metrics to Prometheus."""
    report = client.get_performance_report()
    
    CACHE_HIT_RATE.set(report['engine_performance']['cache_performance']['hit_rate'])
    BANDWIDTH_EFFICIENCY.set(report['bandwidth_metrics']['turns_per_second'])
    
    if 'circuit_breaker_triggered' in report['bandwidth_metrics']:
        CIRCUIT_BREAKER_STATE.set(1 if report['bandwidth_metrics']['circuit_breaker_triggered'] else 0)
    
    return generate_latest()
```

### Health Checks

```python
# health/checks.py
from tboptimizer import ClaudeOptimizedClient, CollaborationContext
import asyncio
import time

class HealthChecker:
    def __init__(self, client: ClaudeOptimizedClient):
        self.client = client
        
    async def check_api_connectivity(self) -> dict:
        """Check API connectivity."""
        try:
            start_time = time.time()
            response = await self.client.collaborate(
                messages=[{"role": "user", "content": "health check"}],
                context=CollaborationContext(session_id="health_check"),
                max_tokens=10
            )
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "optimization_applied": response.optimization_applied
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_cache_performance(self) -> dict:
        """Check cache performance."""
        try:
            report = self.client.get_performance_report()
            cache_performance = report['engine_performance']['cache_performance']
            
            return {
                "status": "healthy" if cache_performance['hit_rate'] > 0.1 else "degraded",
                "hit_rate": cache_performance['hit_rate'],
                "cache_size": cache_performance.get('cache_size', 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_bandwidth_efficiency(self) -> dict:
        """Check bandwidth efficiency."""
        try:
            report = self.client.get_performance_report()
            bandwidth_metrics = report.get('bandwidth_metrics', {})
            
            efficiency = bandwidth_metrics.get('turns_per_second', 0)
            degradation = bandwidth_metrics.get('bandwidth_degradation', 0)
            
            status = "healthy"
            if degradation > 0.5:
                status = "unhealthy"
            elif degradation > 0.2:
                status = "degraded"
            
            return {
                "status": status,
                "efficiency": efficiency,
                "degradation": degradation
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def comprehensive_health_check(self) -> dict:
        """Run comprehensive health check."""
        checks = await asyncio.gather(
            self.check_api_connectivity(),
            self.check_cache_performance(),
            self.check_bandwidth_efficiency(),
            return_exceptions=True
        )
        
        api_health, cache_health, bandwidth_health = checks
        
        # Determine overall status
        all_statuses = [
            api_health.get('status', 'unhealthy'),
            cache_health.get('status', 'unhealthy'),
            bandwidth_health.get('status', 'unhealthy')
        ]
        
        if all(status == 'healthy' for status in all_statuses):
            overall_status = 'healthy'
        elif any(status == 'unhealthy' for status in all_statuses):
            overall_status = 'unhealthy'
        else:
            overall_status = 'degraded'
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "checks": {
                "api_connectivity": api_health,
                "cache_performance": cache_health,
                "bandwidth_efficiency": bandwidth_health
            }
        }
```

### Alerting Rules

```yaml
# alerts.yaml
groups:
- name: tbo-alerts
  rules:
  - alert: TBOHighLatency
    expr: tbo_request_duration_seconds{quantile="0.95"} > 2.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "TBO high latency detected"
      description: "95th percentile latency is {{ $value }}s"

  - alert: TBOCircuitBreakerActive
    expr: tbo_circuit_breaker_active == 1
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "TBO circuit breaker active"
      description: "Circuit breaker has been active for 1 minute"

  - alert: TBOLowCacheHitRate
    expr: tbo_cache_hit_rate < 0.1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "TBO low cache hit rate"
      description: "Cache hit rate is {{ $value }}"

  - alert: TBOBandwidthDegradation
    expr: tbo_bandwidth_efficiency < 0.08
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "TBO bandwidth degradation"
      description: "Bandwidth efficiency is {{ $value }}"
```

## Scaling Strategies

### Horizontal Scaling

```python
# scaling/autoscaler.py
import asyncio
import time
from typing import Dict, Any

class TBOAutoscaler:
    def __init__(self, min_instances=2, max_instances=20):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        
    async def should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """Determine if we should scale up."""
        # Scale up if:
        # 1. CPU usage > 70%
        # 2. Response time > 1.5s
        # 3. Queue length > 50
        
        cpu_usage = metrics.get('cpu_usage', 0)
        avg_response_time = metrics.get('avg_response_time', 0)
        queue_length = metrics.get('queue_length', 0)
        
        return (
            cpu_usage > 0.7 or
            avg_response_time > 1.5 or
            queue_length > 50
        ) and self.current_instances < self.max_instances
    
    async def should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """Determine if we should scale down."""
        # Scale down if:
        # 1. CPU usage < 30%
        # 2. Response time < 0.5s
        # 3. Queue length < 10
        
        cpu_usage = metrics.get('cpu_usage', 0)
        avg_response_time = metrics.get('avg_response_time', 0)
        queue_length = metrics.get('queue_length', 0)
        
        return (
            cpu_usage < 0.3 and
            avg_response_time < 0.5 and
            queue_length < 10
        ) and self.current_instances > self.min_instances
    
    async def scale(self, direction: str):
        """Scale instances up or down."""
        if direction == "up":
            self.current_instances = min(self.max_instances, self.current_instances + 1)
            # Implement scaling logic (Kubernetes, ECS, etc.)
            await self._trigger_scale_up()
        elif direction == "down":
            self.current_instances = max(self.min_instances, self.current_instances - 1)
            await self._trigger_scale_down()
```

### Load Balancing

```nginx
# nginx.conf
upstream tbo_backend {
    least_conn;
    server tbo-1:8000 max_fails=3 fail_timeout=30s;
    server tbo-2:8000 max_fails=3 fail_timeout=30s;
    server tbo-3:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    server_name tbo.yourdomain.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Connection limits
    limit_conn_zone $binary_remote_addr zone=addr:10m;
    limit_conn addr 10;
    
    location / {
        proxy_pass http://tbo_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        proxy_pass http://tbo_backend/health;
        access_log off;
    }
}
```

## Security Considerations

### API Security

```python
# security/auth.py
import jwt
from datetime import datetime, timedelta
from typing import Optional

class APIKeyManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_token(self, user_id: str, permissions: list) -> str:
        """Generate JWT token for API access."""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def rate_limit_key(self, user_id: str) -> str:
        """Generate rate limit key."""
        return f"rate_limit:{user_id}"
```

### Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tbo-network-policy
  namespace: tbo-system
spec:
  podSelector:
    matchLabels:
      app: tbo
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS to LLM APIs
    - protocol: TCP
      port: 6379  # Redis
```

## Backup and Disaster Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

# Backup Redis data
redis-cli --rdb /backup/redis-$(date +%Y%m%d_%H%M%S).rdb

# Backup configuration
kubectl get configmap tbo-config -o yaml > /backup/config-$(date +%Y%m%d_%H%M%S).yaml

# Backup secrets
kubectl get secret api-secrets -o yaml > /backup/secrets-$(date +%Y%m%d_%H%M%S).yaml

# Archive logs
tar -czf /backup/logs-$(date +%Y%m%d_%H%M%S).tar.gz /var/log/tbo/

# Upload to S3
aws s3 sync /backup/ s3://tbo-backups/
```

### Disaster Recovery

```yaml
# disaster-recovery.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dr-runbook
data:
  recovery-steps: |
    1. Verify backup integrity
    2. Restore Redis data from backup
    3. Redeploy TBO services
    4. Restore configuration and secrets
    5. Verify health checks
    6. Resume traffic
  
  rollback-steps: |
    1. Stop current deployment
    2. Deploy previous known-good version
    3. Restore previous configuration
    4. Verify functionality
    5. Update DNS/load balancer
```

## Performance Optimization

### Database Optimization

```python
# optimization/redis.py
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

class OptimizedRedisClient:
    def __init__(self, url: str):
        self.pool = ConnectionPool.from_url(
            url,
            max_connections=100,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
        )
        self.client = redis.Redis(connection_pool=self.pool)
    
    async def get_cached_response(self, key: str) -> Optional[str]:
        """Get cached response with optimized pipeline."""
        pipe = self.client.pipeline()
        pipe.get(key)
        pipe.expire(key, 3600)  # Refresh TTL
        result, _ = await pipe.execute()
        return result
    
    async def set_cached_response(self, key: str, value: str, ttl: int = 3600):
        """Set cached response with pipeline."""
        pipe = self.client.pipeline()
        pipe.set(key, value, ex=ttl)
        pipe.incr(f"cache_stats:sets")
        await pipe.execute()
```

### Connection Pooling

```python
# optimization/connections.py
import aiohttp
import asyncio
from typing import Optional

class OptimizedHTTPClient:
    def __init__(self):
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=20,  # Per-host connection limit
            keepalive_timeout=300,  # Keep connections alive
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,  # Total timeout
            connect=5,  # Connection timeout
            sock_read=10  # Socket read timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'TBO/1.0'}
        )
    
    async def close(self):
        await self.session.close()
```

## Cost Optimization

### Resource Management

```yaml
# resources.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tbo-quota
  namespace: tbo-system
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    persistentvolumeclaims: "10"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: tbo-limits
  namespace: tbo-system
spec:
  limits:
  - default:
      cpu: "500m"
      memory: "1Gi"
    defaultRequest:
      cpu: "250m"
      memory: "512Mi"
    type: Container
```

### Cost Monitoring

```python
# cost/monitor.py
class CostMonitor:
    def __init__(self):
        self.api_costs = {
            'claude': 0.008,  # per 1k tokens
            'gpt': 0.002     # per 1k tokens
        }
    
    def calculate_request_cost(self, provider: str, tokens: int) -> float:
        """Calculate cost per request."""
        cost_per_1k = self.api_costs.get(provider, 0)
        return (tokens / 1000) * cost_per_1k
    
    def optimize_model_selection(self, task_complexity: int) -> str:
        """Select cost-optimal model based on task."""
        if task_complexity <= 2:
            return "gpt-3.5-turbo"  # Lower cost
        elif task_complexity <= 4:
            return "claude-3-sonnet"  # Balanced
        else:
            return "claude-3-opus"  # High quality
```

---

This deployment guide provides enterprise-ready patterns for scaling TBO in production environments. Adapt the configurations to your specific infrastructure and requirements.