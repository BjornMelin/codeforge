# ADR-012: Advanced Embeddings

**Status**: Accepted

**Context**: Phase 2 of CodeForge AI requires enhanced retrieval precision beyond dense embeddings alone. While BGE-M3 provides excellent semantic understanding, sparse embeddings capture exact lexical matches critical for technical documentation, API names, and specific code patterns that dense embeddings might miss.

**Decision**: Implement hybrid sparse+dense embedding architecture using SPLADE for sparse embeddings alongside existing BGE-M3 dense embeddings via sentence-transformers v5.0.0+, with weighted fusion for optimal retrieval accuracy.

**Consequences**:

- Positive: Significant improvement in retrieval precision, better handling of technical terminology, improved exact match capabilities, maintains semantic understanding
- Negative: Increased computational overhead, higher storage requirements, complexity in result fusion, need for careful weight tuning

## Architecture Overview

### Hybrid Embedding Strategy

- **Dense Embeddings**: BGE-M3 for semantic similarity and conceptual understanding
- **Sparse Embeddings**: SPLADE for exact lexical matching and technical terminology
- **Fusion Approach**: Weighted combination optimizing for both semantic relevance and exact matches
- **Content-Aware Processing**: Different embedding strategies based on content type

### Retrieval Enhancement Patterns

- **Technical Documentation**: Emphasize sparse embeddings for API references and exact syntax
- **Code Patterns**: Balance dense for algorithmic concepts, sparse for specific function names
- **Error Messages**: Prioritize sparse matching for exact error text and stack traces
- **Conceptual Queries**: Favor dense embeddings for broad architectural discussions

### Performance Optimization

- **Quantization**: Maintain int8 quantization for memory efficiency
- **Selective Processing**: Apply appropriate embedding strategy based on query type
- **Caching Strategy**: Cache frequently accessed embeddings with content-aware TTL
- **Batch Processing**: Optimize embedding generation for bulk operations

## Embedding Architecture

### Dual-Track Processing

- **Dense Track**: BGE-M3 processing for semantic vectors (384D code, 768D docs)
- **Sparse Track**: SPLADE processing for lexical term weights
- **Fusion Layer**: Intelligent combination based on query characteristics
- **Quality Assessment**: Real-time evaluation of retrieval effectiveness

### Query-Adaptive Routing

- **Semantic Queries**: Emphasize dense embeddings for conceptual understanding
- **Exact Match Queries**: Prioritize sparse embeddings for precise terminology
- **Hybrid Queries**: Balanced approach for complex multi-faceted requests
- **Domain Detection**: Automatic classification of technical vs conceptual content

### Implementation Architecture

```pseudocode
HybridEmbeddingEngine {
  denseModel: BGE-M3<int8_quantized>
  sparseModel: SPLADE
  fusionWeights: AdaptiveWeights
  queryClassifier: QueryAnalyzer
  
  generateEmbeddings(content, contentType) -> HybridEmbedding {
    dense = denseModel.encode(content, quantize=true)
    sparse = sparseModel.encode(content)
    
    return HybridEmbedding(dense, sparse, contentType)
  }
  
  search(query, corpus) -> RankedResults {
    queryType = queryClassifier.classify(query)
    weights = fusionWeights.getWeights(queryType)
    
    denseResults = denseSearch(query, corpus)
    sparseResults = sparseSearch(query, corpus)
    
    return fuseResults(denseResults, sparseResults, weights)
  }
}

AdaptiveWeightingSystem {
  baseWeights: Map<QueryType, WeightPair>
  performanceHistory: FeedbackTracker
  
  getWeights(queryType) -> WeightPair {
    base = baseWeights[queryType]
    adjustment = performanceHistory.getOptimalAdjustment(queryType)
    return adjustWeights(base, adjustment)
  }
}
```

## Content-Specific Strategies

### Code Content Processing

- **Function Names**: High sparse weight for exact matching
- **Variable Names**: Balanced approach for both exact and semantic matches
- **Comments**: Dense embedding emphasis for conceptual understanding
- **Documentation Strings**: Hybrid approach with moderate sparse weighting

### Technical Documentation

- **API References**: Sparse emphasis for exact method and parameter names
- **Conceptual Guides**: Dense emphasis for understanding and explanation
- **Configuration Files**: Sparse priority for exact key-value matching
- **Error Messages**: High sparse weight for exact error text matching

## Success Criteria

### Retrieval Accuracy Improvements

- **Overall Precision**: 15% improvement over dense-only baseline (target: 95% relevance)
- **Technical Term Matching**: 30% improvement for exact API and function name queries
- **Semantic Understanding**: Maintain 95% of dense embedding semantic capabilities
- **Hybrid Query Performance**: 20% improvement for complex multi-faceted queries
- **Domain-Specific Accuracy**: 25% improvement for technical documentation retrieval

### Performance Targets

- **Embedding Generation**: <200ms for hybrid embedding creation (vs 100ms dense-only)
- **Search Latency**: <800ms for hybrid search (vs 500ms dense-only)
- **Memory Overhead**: <50% increase in storage requirements
- **Fusion Processing**: <100ms for result combination and ranking
- **Cache Hit Rate**: >70% for frequently accessed embeddings

### Quality Metrics

- **False Positive Rate**: <5% irrelevant results in top 10 (vs 8% dense-only)
- **Coverage**: >98% of technical terms properly indexed in sparse embeddings
- **Consistency**: <10% variance in results for semantically equivalent queries
- **Adaptivity**: Weights improve 15% per month through performance feedback

### Resource Efficiency

- **Computational Overhead**: <2x processing time vs dense-only approach
- **Storage Optimization**: Efficient sparse representation with <3x storage increase
- **Memory Usage**: <4GB total for hybrid embedding processing
- **Network Efficiency**: Optimized transfer of sparse embedding data

## Implementation Strategy

### Phase 2A: SPLADE Integration (Week 1-3)

- Implement SPLADE sparse embedding generation
- Add basic sparse search capabilities alongside existing dense search
- Test hybrid fusion with simple technical queries

### Phase 2B: Adaptive Fusion (Week 4-6)

- Implement query classification and adaptive weighting system
- Add content-type-aware processing strategies
- Performance optimization and caching implementation

### Phase 2C: Production Optimization (Week 7-9)

- Fine-tune fusion weights based on production query patterns
- Add comprehensive monitoring and quality metrics
- Load testing and capacity planning for production deployment

### Future Enhancements

- Machine learning-based fusion weight optimization
- Dynamic embedding strategy selection based on corpus characteristics
- Integration with federated search across multiple knowledge bases
