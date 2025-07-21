# ADR-015: Enhanced Scalability (Phase 2)

**Status**: Proposed (for Phase 2)  

**Date**: 2025-07-20  

**Deciders**: CodeForge AI Team  

## Context

Scale Phase 1 Docker to >100 agents. Phase 1's Docker Compose setup works well for MVP with 3-10 agents, but Phase 2 needs to support 500+ agents with enterprise-grade orchestration and advanced tooling capabilities.

## Problem Statement

Phase 1 local deployment needs hyperscale orchestration opt-in. Requirements include:

- Support for 500+ concurrent agents without performance degradation

- Enterprise-grade orchestration and monitoring capabilities

- Advanced web scraping and tooling for complex workflows

- Opt-in complexity to avoid overwhelming simple deployments

- Seamless migration path from Phase 1 Docker setup

## Decision

**Kubernetes opt-in for orchestration** with ZenRows web alternative and enhanced tooling for enterprise scalability.

## Alternatives Considered

| Approach | Pros | Cons | Score |
|----------|------|------|-------|
| **Kubernetes Opt-in** | Industry standard, proven scalability, rich ecosystem | Complexity, learning curve | **8.0** |
| Stick to Docker | Sufficient for MVP, simpler | Limited scalability, manual orchestration | 8.0 |
| Docker Swarm | Simpler than K8s, Docker-native | Less ecosystem, limited enterprise features | 7.5 |
| Cloud-specific solutions | Managed services, easy setup | Vendor lock-in, higher costs | 7.3 |

## Rationale

- **Future-proof scalability (8.0)**: Supports growth to enterprise scale

- **Industry standard**: Kubernetes is the de facto orchestration platform

- **Opt-in complexity**: Doesn't impact simple deployments

- **Rich ecosystem**: Access to monitoring, scaling, and management tools

## Consequences

### Positive

- Massive scalability for enterprise deployments

- Professional orchestration with auto-scaling and health monitoring

- Rich ecosystem of tools and integrations

- Future-proof architecture for continued growth

### Negative

- Significant complexity increase for enterprise deployments

- Learning curve and operational overhead

- Infrastructure requirements and costs

### Neutral

- Opt-in design to avoid disrupting simple deployments

- Migration tooling to ease transition from Docker

## Implementation Notes

### Kubernetes Architecture
```yaml

# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: codeforge-ai
  labels:
    app: codeforge
    environment: production

---

# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: codeforge-config
  namespace: codeforge-ai
data:
  PHASE: "2"
  MAX_AGENTS: "500"
  REDIS_HOST: "redis-service"
  NEO4J_HOST: "neo4j-service"
  QDRANT_HOST: "qdrant-service"
  OPENROUTER_API_KEY: ""  # Set via secret
  FEDERATION_ENABLED: "true"
  SCALABILITY_MODE: "kubernetes"

---

# kubernetes/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: codeforge-secrets
  namespace: codeforge-ai
type: Opaque
data:
  # Base64 encoded secrets
  OPENROUTER_API_KEY: ""
  TAVILY_API_KEY: ""
  EXA_API_KEY: ""
  ZENROWS_API_KEY: ""
```

### Core Services Deployment
```yaml

# kubernetes/redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: codeforge-ai
spec:
  replicas: 3  # Redis cluster for HA
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7.2-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: codeforge-ai
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP

---

# kubernetes/neo4j-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neo4j
  namespace: codeforge-ai
spec:
  replicas: 1  # Single instance for simplicity, can be clustered
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.15
        ports:
        - containerPort: 7474
        - containerPort: 7687
        env:
        - name: NEO4J_AUTH
          value: "neo4j/codeforge-password"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: neo4j-data
          mountPath: /data
      volumes:
      - name: neo4j-data
        persistentVolumeClaim:
          claimName: neo4j-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: neo4j-service
  namespace: codeforge-ai
spec:
  selector:
    app: neo4j
  ports:
  - name: http
    port: 7474
    targetPort: 7474
  - name: bolt
    port: 7687
    targetPort: 7687
  type: ClusterIP

---

# kubernetes/qdrant-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
  namespace: codeforge-ai
spec:
  replicas: 2  # Multiple instances for load distribution
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.7.3
        ports:
        - containerPort: 6333
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: qdrant-data
          mountPath: /qdrant/storage
      volumes:
      - name: qdrant-data
        persistentVolumeClaim:
          claimName: qdrant-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service
  namespace: codeforge-ai
spec:
  selector:
    app: qdrant
  ports:
  - port: 6333
    targetPort: 6333
  type: ClusterIP
```

### Agent Orchestration
```yaml

# kubernetes/agent-orchestrator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
  namespace: codeforge-ai
spec:
  replicas: 3  # Multiple orchestrators for HA
  selector:
    matchLabels:
      app: agent-orchestrator
  template:
    metadata:
      labels:
        app: agent-orchestrator
    spec:
      containers:
      - name: orchestrator
        image: codeforge/agent-orchestrator:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: codeforge-config
              key: REDIS_HOST
        - name: NEO4J_HOST
          valueFrom:
            configMapKeyRef:
              name: codeforge-config
              key: NEO4J_HOST
        - name: QDRANT_HOST
          valueFrom:
            configMapKeyRef:
              name: codeforge-config
              key: QDRANT_HOST
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: codeforge-secrets
              key: OPENROUTER_API_KEY
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

---

# kubernetes/agent-workers.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-workers
  namespace: codeforge-ai
spec:
  replicas: 100  # Start with 100, can scale to 500+
  selector:
    matchLabels:
      app: agent-worker
  template:
    metadata:
      labels:
        app: agent-worker
    spec:
      containers:
      - name: worker
        image: codeforge/agent-worker:latest
        env:
        - name: ORCHESTRATOR_HOST
          value: "agent-orchestrator-service"
        - name: WORKER_TYPE
          value: "general"  # Can be specialized: coding, analysis, debate
        - name: MAX_CONCURRENT_TASKS
          value: "3"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"

---

# kubernetes/hpa.yaml - Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-workers-hpa
  namespace: codeforge-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-workers
  minReplicas: 100
  maxReplicas: 500
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

### Enhanced Web Scraping (ZenRows Integration)
```python
from typing import List, Dict, Optional, Any
import asyncio
import httpx
from dataclasses import dataclass

@dataclass
class ZenRowsConfig:
    api_key: str
    proxy_country: str = "US"
    js_render: bool = True
    wait_time: int = 3000
    block_resources: List[str] = None
    
    def __post_init__(self):
        if self.block_resources is None:
            self.block_resources = ["image", "media", "font"]

class EnhancedWebScraper:
    """Phase 2 enhanced web scraping with ZenRows for complex sites"""
    
    def __init__(self, zenrows_config: Optional[ZenRowsConfig] = None):
        self.zenrows_config = zenrows_config
        self.zenrows_enabled = zenrows_config is not None
        
        # Standard scraping for simple sites
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        # Rate limiting
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        
    async def scrape_advanced(self, urls: List[str], 
                            complexity_level: str = "standard") -> List[Dict[str, Any]]:
        """Advanced scraping with complexity-based routing"""
        
        results = []
        
        for url in urls:
            await self.rate_limiter.acquire()
            
            try:
                if complexity_level == "complex" and self.zenrows_enabled:
                    result = await self._zenrows_scrape(url)
                else:
                    result = await self._standard_scrape(url)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                results.append({"url": url, "error": str(e), "content": ""})
        
        return results
    
    async def _zenrows_scrape(self, url: str) -> Dict[str, Any]:
        """Advanced scraping using ZenRows for complex sites"""
        
        zenrows_url = "https://api.zenrows.com/v1/"
        
        params = {
            "url": url,
            "apikey": self.zenrows_config.api_key,
            "js_render": str(self.zenrows_config.js_render).lower(),
            "wait": self.zenrows_config.wait_time,
            "block_resources": ",".join(self.zenrows_config.block_resources),
            "proxy_country": self.zenrows_config.proxy_country
        }
        
        async with self.http_client as client:
            response = await client.get(zenrows_url, params=params)
            response.raise_for_status()
            
            return {
                "url": url,
                "content": response.text,
                "status_code": response.status_code,
                "method": "zenrows",
                "js_rendered": True
            }
    
    async def _standard_scrape(self, url: str) -> Dict[str, Any]:
        """Standard HTTP scraping for simple sites"""
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        async with self.http_client as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            return {
                "url": url,
                "content": response.text,
                "status_code": response.status_code,
                "method": "standard",
                "js_rendered": False
            }
    
    async def detect_scraping_complexity(self, url: str) -> str:
        """Detect if a site needs advanced scraping"""
        
        # Simple heuristics for complexity detection
        complex_indicators = [
            "cloudflare", "captcha", "javascript", "react", "angular", "vue",
            "single-page", "spa", "ajax", "dynamic"
        ]
        
        try:
            # Quick HEAD request to check headers and response
            async with self.http_client as client:
                response = await client.head(url, timeout=5.0)
                
                # Check headers for complexity indicators
                server_header = response.headers.get("server", "").lower()
                
                if any(indicator in server_header for indicator in complex_indicators):
                    return "complex"
                
                # Check if content-type suggests SPA
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    return "complex"
                
                return "standard"
                
        except Exception:
            # If we can't determine, assume complex for safety
            return "complex"
```

### Monitoring and Observability
```yaml

# kubernetes/monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: codeforge-ai
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'codeforge-agents'
      static_configs:
      - targets: ['agent-orchestrator-service:8080']
    - job_name: 'codeforge-workers'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - codeforge-ai
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: agent-worker

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: codeforge-ai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        resources:
          requests:
            memory: "512Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config

---

# Grafana for visualization
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: codeforge-ai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "codeforge-admin"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
```

### Migration Tooling
```python
class DockerToKubernetesMigration:
    """Tool to migrate from Phase 1 Docker to Phase 2 Kubernetes"""
    
    def __init__(self, docker_compose_path: str, k8s_output_dir: str):
        self.docker_compose_path = docker_compose_path
        self.k8s_output_dir = k8s_output_dir
        
    async def migrate_configuration(self) -> bool:
        """Migrate Docker Compose configuration to Kubernetes manifests"""
        
        try:
            # Parse existing Docker Compose
            compose_config = self._parse_docker_compose()
            
            # Generate Kubernetes manifests
            k8s_manifests = self._generate_k8s_manifests(compose_config)
            
            # Write manifests to output directory
            await self._write_manifests(k8s_manifests)
            
            # Generate migration guide
            await self._generate_migration_guide()
            
            logger.info(f"Migration completed. Kubernetes manifests written to {self.k8s_output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _parse_docker_compose(self) -> Dict[str, Any]:
        """Parse existing Docker Compose configuration"""
        import yaml
        
        with open(self.docker_compose_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_k8s_manifests(self, compose_config: Dict[str, Any]) -> Dict[str, str]:
        """Generate Kubernetes manifests from Docker Compose"""
        
        manifests = {}
        
        for service_name, service_config in compose_config.get('services', {}).items():
            # Generate Deployment
            deployment = self._create_deployment_manifest(service_name, service_config)
            manifests[f"{service_name}-deployment.yaml"] = deployment
            
            # Generate Service if ports are exposed
            if 'ports' in service_config:
                service = self._create_service_manifest(service_name, service_config)
                manifests[f"{service_name}-service.yaml"] = service
        
        return manifests
    
    async def _write_manifests(self, manifests: Dict[str, str]):
        """Write Kubernetes manifests to files"""
        
        import os
        os.makedirs(self.k8s_output_dir, exist_ok=True)
        
        for filename, content in manifests.items():
            filepath = os.path.join(self.k8s_output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
    
    async def _generate_migration_guide(self):
        """Generate step-by-step migration guide"""
        
        guide = """

# CodeForge AI Migration Guide: Docker to Kubernetes

## Prerequisites
1. Install kubectl and configure access to your Kubernetes cluster
2. Install Helm (optional, for package management)
3. Ensure cluster has sufficient resources (CPU: 10+ cores, Memory: 32+ GB)

## Migration Steps

### 1. Backup Current Data
```bash

# Backup Docker volumes
docker run --rm -v codeforge_neo4j_data:/data -v $(pwd):/backup busybox tar czf /backup/neo4j-backup.tar.gz /data
docker run --rm -v codeforge_qdrant_data:/data -v $(pwd):/backup busybox tar czf /backup/qdrant-backup.tar.gz /data
```

### 2. Deploy to Kubernetes
```bash

# Create namespace
kubectl create namespace codeforge-ai

# Apply configurations
kubectl apply -f kubernetes/

# Wait for deployments
kubectl wait --for=condition=available --timeout=600s deployment --all -n codeforge-ai
```

### 3. Restore Data
```bash

# Restore data to Kubernetes persistent volumes

# (Implementation depends on your storage provider)
```

### 4. Validate Migration
```bash

# Check pod status
kubectl get pods -n codeforge-ai

# Check services
kubectl get services -n codeforge-ai

# Test agent orchestrator
kubectl port-forward service/agent-orchestrator-service 8080:8080 -n codeforge-ai
```

## Scaling Commands
```bash

# Scale agents up
kubectl scale deployment agent-workers --replicas=200 -n codeforge-ai

# Scale agents down
kubectl scale deployment agent-workers --replicas=50 -n codeforge-ai

# Auto-scaling is handled by HPA
```

## Monitoring
```bash

# Port forward to Grafana
kubectl port-forward service/grafana 3000:3000 -n codeforge-ai

# Access at http://localhost:3000 (admin/codeforge-admin)
```
        """
        
        guide_path = os.path.join(self.k8s_output_dir, "MIGRATION_GUIDE.md")
        with open(guide_path, 'w') as f:
            f.write(guide)
```

## Performance Targets

| Metric | Phase 1 (Docker) | Phase 2 (Kubernetes) | Improvement |
|--------|------------------|----------------------|-------------|
| Max Concurrent Agents | 10 | 500+ | 50x |
| Auto-scaling Response | Manual | <2 minutes | Automated |
| High Availability | Single instance | Multi-replica | Enterprise-grade |
| Monitoring Coverage | Basic logs | Full observability | Professional |
| Deployment Complexity | Simple | Managed complexity | Opt-in |

## Cost Considerations

### Resource Requirements

- **Small deployment (100 agents)**: 8 CPU cores, 16GB RAM

- **Medium deployment (250 agents)**: 20 CPU cores, 40GB RAM

- **Large deployment (500 agents)**: 40 CPU cores, 80GB RAM

### Cloud Costs (Estimated)

- **AWS EKS**: $150-300/month for small, $400-800/month for large

- **GCP GKE**: $120-250/month for small, $350-700/month for large

- **Azure AKS**: $130-270/month for small, $380-750/month for large

## Rollout Strategy

### Phase 2A: Foundation Setup

- Deploy Kubernetes manifests

- Migrate core services (Redis, Neo4j, Qdrant)

- Test basic agent orchestration

### Phase 2B: Scaling Validation  

- Scale to 100+ agents

- Enable auto-scaling

- Validate performance under load

### Phase 2C: Enterprise Features

- Deploy monitoring stack

- Enable ZenRows for complex scraping

- Federation integration testing

## Related Decisions

- ADR-010: Task Management System

- ADR-002: Database and Memory System

- ADR-007: Web Search Integration

## Monitoring

- Agent deployment success rates and scaling metrics

- Resource utilization and cost optimization

- Migration success and rollback procedures

- Performance comparison vs Phase 1 Docker

- Enterprise feature adoption and effectiveness
