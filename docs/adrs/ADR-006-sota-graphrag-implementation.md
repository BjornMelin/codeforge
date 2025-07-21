# ADR-006: State-of-the-Art GraphRAG+ Implementation

**Status**: Accepted

**Context**: CodeForge AI requires advanced retrieval capabilities beyond standard RAG to handle complex code relationships and architectural decisions in Phase 1. Must achieve 30% accuracy improvement over baseline RAG while supporting Phase 2 multi-modal extensions. Current RAG approaches fail with code dependencies, architectural patterns, and cross-file relationships critical for autonomous development.

**Decision**: GraphRAG+ hybrid approach combining knowledge graphs (Neo4j v5.28.1+) with vector similarity (Qdrant v1.15.0+) using BGE-M3 embeddings with int8 quantization (sentence-transformers v5.0.0+), extending with SPLADE sparse embeddings in Phase 2.

**Consequences**:

- Positive: 30% accuracy improvement over baseline RAG, captures code relationships and architectural patterns, supports multi-hop reasoning, natural extension to sparse+dense hybrid in Phase 2
- Negative: Increased complexity over simple vector search, higher computational requirements, need for sophisticated graph construction and maintenance

## Architecture Overview

### Hybrid Retrieval Design

- **Knowledge Graph Layer**: Code relationships, dependencies, architectural patterns (Neo4j)
- **Vector Similarity Layer**: Semantic search across code and documentation (Qdrant)
- **Embedding Strategy**: BGE-M3 with content-aware dimensions (384 code, 768 docs, 256 functions)
- **Query Processing**: Hybrid search combining graph traversal with vector similarity

### Graph Knowledge Representation

- **Code Entities**: Functions, classes, modules, files as graph nodes
- **Relationships**: Function calls, imports, inheritance, dependencies as edges
- **Architectural Patterns**: Design patterns, system architecture as higher-level nodes
- **Semantic Connections**: Code similarity, conceptual relationships via embeddings

### Retrieval Strategy Requirements

- **Multi-hop Reasoning**: Traverse relationships to find relevant context 2-3 hops away
- **Context Ranking**: Combine graph centrality with vector similarity scores
- **Query Expansion**: Use graph relationships to expand search scope
- **Result Fusion**: Merge graph and vector results with weighted scoring

## Phase 2 Extensions

### Sparse+Dense Hybrid Embeddings

- **Dense Embeddings**: BGE-M3 for semantic similarity (existing)
- **Sparse Embeddings**: SPLADE for lexical/keyword matching (new)
- **Fusion Strategy**: Weighted combination of sparse and dense scores
- **Performance Target**: Additional 15% accuracy improvement over dense-only

### Multi-Modal Integration

- **Vision Embeddings**: CLIP for UI/code screenshot analysis
- **Cross-Modal Search**: Find code relevant to visual components
- **Architectural Diagrams**: Link diagrams to corresponding code structures

### Implementation Architecture

```pseudocode
GraphRAGPlus {
  graphDB: Neo4j<code_relationships>
  vectorDB: Qdrant<semantic_embeddings>
  embeddingModel: BGE-M3<int8_quantized>
  
  hybridSearch(query) -> RankedResults {
    vectorResults = vectorDB.search(query.embedding)
    graphResults = graphDB.traverse(vectorResults.topK(5))
    return fuseResults(vectorResults, graphResults)
  }
  
  storeCodeKnowledge(codeData) {
    relationships = extractRelationships(codeData)
    graphDB.store(relationships)
    embedding = embeddingModel.encode(codeData)
    vectorDB.store(embedding)
  }
}

Phase2Extensions {
  sparseEmbeddings: SPLADE
  visionEmbeddings: CLIP
  
  hybridEmbeddingSearch(query) -> Results {
    dense = BGE-M3.encode(query)
    sparse = SPLADE.encode(query)
    return weightedFusion(dense, sparse, weights=[0.7, 0.3])
  }
}
```

## Success Criteria

### Accuracy Targets

- **Phase 1**: 30% improvement over baseline RAG (target: 85% relevance score)
- **Phase 2**: Additional 15% improvement with sparse+dense hybrid (target: 95% relevance)
- **Multi-hop Retrieval**: >80% accuracy for 2-hop relationship queries
- **Code Context**: >90% accuracy for finding related functions/classes

### Performance Requirements

- **Query Latency**: <500ms for hybrid search (vector + graph traversal)
- **Index Size**: <2GB for 10K code files with int8 quantization
- **Memory Usage**: <1GB for active embeddings and graph cache
- **Throughput**: 100+ queries/second sustained load

### Quality Metrics

- **Relevance Score**: >85% for retrieved code context in Phase 1
- **Coverage**: >95% of code relationships captured in knowledge graph
- **Precision**: >80% for graph traversal results (reduce noise)
- **Recall**: >90% for finding related architectural components

### Scalability Targets

- **Code Base Size**: 10K+ files in Phase 1, 100K+ files in Phase 2
- **Graph Complexity**: 1M+ nodes and 10M+ edges in Phase 2
- **Concurrent Queries**: 50+ simultaneous searches without degradation
- **Update Latency**: <1 minute to index new code changes

## Implementation Strategy

### Phase 1A: Core GraphRAG (Week 1-2)

- Neo4j + Qdrant integration with BGE-M3 embeddings
- Basic graph construction from code analysis
- Hybrid search with simple result fusion

### Phase 1B: Advanced Retrieval (Week 3-4)

- Multi-hop graph traversal optimization
- Sophisticated result ranking and fusion
- Performance tuning for 30% accuracy target

### Phase 1C: Production Optimization (Week 5-6)

- Index optimization and caching strategies
- Real-time graph updates for code changes
- Monitoring and quality metrics collection

### Phase 2 Preparation

- SPLADE integration architecture design
- Multi-modal embedding framework
- Federated knowledge graph considerations
