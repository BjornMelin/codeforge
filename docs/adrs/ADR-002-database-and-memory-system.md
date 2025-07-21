# ADR-002: Database and Memory System

**Status**: Accepted

**Context**: CodeForge AI needs scalable memory architecture for RAG and state management in Phase 1, extensible to Phase 2 multi-modal and federated requirements. Must balance local deployment with accuracy gains while maintaining cost efficiency. Requires graph-based knowledge representation, vector similarity search, and fast caching with 30% accuracy improvement over baseline RAG.

**Decision**: Hybrid architecture using Neo4j v5.28.1+ (graphs) + Qdrant v1.15.0+ (vectors) + Redis v6.0.0+ (Pub/Sub/caching) in Phase 1, with SQLite toggle for lightweight deployments.

**Consequences**:

- Positive: Enables GraphRAG+ with 30% accuracy boost, supports both graph and vector operations, flexible deployment options, best performance for each data type, proven scalability
- Negative: Multiple services increase deployment complexity, higher resource requirements, synchronization overhead between systems

## Architecture Overview

### Multi-Database Strategy

- **Neo4j (Graph Database)**: Code relationships, dependencies, architectural patterns
- **Qdrant (Vector Database)**: Semantic embeddings with content-aware dimensions
- **Redis (Cache/PubSub)**: Fast caching, agent coordination, task queues
- **SQLite (Lightweight Mode)**: Simplified deployment for resource-constrained environments

### Data Storage Patterns

- **Code Entities**: Functions, classes, modules as graph nodes with vector embeddings
- **Relationships**: Function calls, imports, inheritance as graph edges
- **Embeddings**: BGE-M3 with int8 quantization (384D code, 768D docs, 256D functions)
- **Cache Strategy**: Hierarchical caching with TTL management and invalidation patterns

### Deployment Flexibility

- **Full Mode**: All three databases for maximum performance and accuracy
- **Lightweight Mode**: SQLite + in-memory Qdrant for minimal resource usage
- **Auto-Fallback**: Automatic degradation to lightweight mode if services unavailable
- **Horizontal Scaling**: Redis Cluster support for Phase 2 distributed deployments

## GraphRAG+ Integration

### Hybrid Search Architecture

- **Vector Similarity**: Semantic search across code and documentation
- **Graph Traversal**: Multi-hop relationship exploration (2-3 hops)
- **Result Fusion**: Weighted combination of vector and graph scores
- **Query Expansion**: Graph relationships inform search scope

### Performance Optimization

- **Embedding Quantization**: int8 reduces memory usage by 75% with minimal accuracy loss
- **Connection Pooling**: Optimized database connections for concurrent access
- **Batch Operations**: Efficient bulk updates and queries
- **Cache Warming**: Proactive caching of frequently accessed data

### Implementation Architecture

```pseudocode
DatabaseManager {
  graphDB: Neo4j
  vectorDB: Qdrant  
  cacheDB: Redis
  lightweightMode: Boolean
  
  storeCodeKnowledge(codeData) {
    relationships = extractRelationships(codeData)
    graphDB.store(relationships)
    
    embedding = BGE-M3.encode(codeData, quantize=int8)
    vectorDB.store(embedding)
    
    cacheDB.invalidate(relatedKeys)
  }
  
  hybridSearch(query) -> Results {
    vector_results = vectorDB.search(query.embedding)
    graph_results = graphDB.traverse(vector_results.topK(5))
    return fuseResults(vector_results, graph_results, weights=[0.7, 0.3])
  }
}

LightweightFallback {
  sqliteDB: SQLite
  memoryVectorDB: Qdrant(":memory:")
  
  // Simplified operations for resource-constrained deployments
}
```

## Success Criteria

### Performance Targets

- **Search Accuracy**: 30% improvement over baseline RAG (target: 85% relevance score)
- **Query Latency**: <500ms for hybrid search (vector + graph traversal)
- **Embedding Storage**: <2GB for 10K code files with efficient quantization
- **Cache Hit Rate**: >70% for repeated queries
- **Memory Usage**: <1GB total for database connections and caches

### Reliability Metrics

- **Database Uptime**: >99.5% availability for full mode
- **Data Consistency**: Zero data loss during normal operations
- **Failover Time**: <30s to lightweight mode if full mode fails
- **Backup Success**: Daily automated backups with <5 minute recovery time
- **Concurrent Access**: Support 50+ simultaneous queries without degradation

### Deployment Metrics

- **Resource Efficiency**: <4GB RAM for full mode, <1GB for lightweight mode
- **Startup Time**: <2 minutes for full initialization
- **Auto-Recovery**: 95% success rate for automatic service recovery
- **Scaling Readiness**: Phase 2 federated support with minimal architectural changes

## Implementation Strategy

### Phase 1A: Core Database Setup (Week 1-2)

- Deploy Neo4j, Qdrant, Redis stack with basic configuration
- Implement basic GraphRAG+ with vector + graph hybrid search
- Validate 30% accuracy improvement over baseline RAG

### Phase 1B: Performance Optimization (Week 3-4)

- Add caching layer and optimize query patterns
- Implement automatic lightweight mode fallback
- Load testing and memory optimization for target metrics

### Phase 1C: Production Hardening (Week 5-6)

- Add comprehensive monitoring and alerting
- Implement automated backup and recovery procedures
- Documentation and operational runbooks for deployment
