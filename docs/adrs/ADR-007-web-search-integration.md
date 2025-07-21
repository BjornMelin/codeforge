# ADR-007: Web Search Integration

**Status**: Accepted

**Context**: CodeForge AI needs real-time web search capabilities for RAG misses and current events in Phase 1, essential for autonomous development when local knowledge is insufficient. Must balance cost, latency, and quality while maintaining <2000ms search latency target and handling rate limits gracefully.

**Decision**: Tavily v0.7.10+ as primary search engine with Exa fallback for specialized queries, extending with ZenRows for advanced scraping in Phase 2.

**Consequences**:

- Positive: High-quality search results, built-in content processing, reliable fallback strategy, specialized tool for different query types, proven API stability
- Negative: External service dependency, potential cost scaling, rate limit management complexity, need for intelligent query routing

## Architecture Overview

### Multi-Provider Strategy

- **Primary Search**: Tavily for general web search with content extraction
- **Specialized Search**: Exa for research papers, technical documentation, and developer content
- **Fallback Chain**: Automatic failover from Tavily → Exa → cached results
- **Advanced Scraping**: ZenRows integration for complex sites (Phase 2)

### Search Trigger Patterns

- **RAG Miss Detection**: When GraphRAG+ confidence score <0.6
- **Current Event Queries**: Date-sensitive information and breaking news
- **Technical Documentation**: Framework updates, API changes, best practices
- **Error Resolution**: Stack traces, error messages, troubleshooting guides

### Content Processing Pipeline

- **Query Analysis**: Intent classification and provider routing
- **Result Filtering**: Content quality assessment and relevance scoring
- **Content Extraction**: Clean text extraction with metadata preservation
- **Cache Management**: Intelligent caching with TTL based on content type

## Integration Strategy

### Provider Selection Logic

- **Tavily Use Cases**: General coding questions, tutorials, documentation updates
- **Exa Use Cases**: Research papers, academic content, deep technical analysis
- **Query Routing**: Automatic provider selection based on query characteristics
- **Cost Optimization**: Prefer lower-cost providers when quality difference is minimal

### Rate Limit Management

- **Token Bucket Algorithm**: Smooth rate limiting with burst capacity
- **Provider Rotation**: Distribute load across available providers
- **Graceful Degradation**: Fall back to cached results when rate limited
- **Priority Queuing**: Critical queries bypass normal rate limits

### Implementation Architecture

```pseudocode
WebSearchOrchestrator {
  primaryProvider: TavilyClient
  fallbackProvider: ExaClient
  scraperProvider: ZenRowsClient  // Phase 2
  contentCache: CacheManager
  rateLimiter: TokenBucket
  
  search(query, context) -> SearchResults {
    if (isRAGMiss(context) || isCurrentEvent(query)) {
      provider = selectProvider(query)
      
      try {
        results = provider.search(query, filters)
        processedResults = extractAndFilter(results)
        cacheResults(query, processedResults)
        return processedResults
      } catch (RateLimitError) {
        return fallbackSearch(query)
      }
    }
    
    return getCachedResults(query)
  }
  
  selectProvider(query) -> SearchProvider {
    if (isResearchQuery(query)) return fallbackProvider
    if (isGeneralQuery(query)) return primaryProvider
    return primaryProvider  // default
  }
}

ContentProcessor {
  extractCleanText(rawResults) -> ProcessedContent
  assessRelevance(content, query) -> RelevanceScore
  filterByQuality(results) -> FilteredResults
}
```

## Success Criteria

### Performance Targets

- **Search Latency**: <2000ms for web search requests (95th percentile)
- **Cache Hit Rate**: >40% for repeated technical queries
- **Relevance Score**: >80% relevance for search results vs user intent
- **Availability**: >99% uptime with graceful fallback handling
- **Cost Efficiency**: <$50/month for typical usage patterns (1000 searches/day)

### Quality Metrics

- **RAG Miss Resolution**: 70% of RAG misses resolved with web search
- **Content Freshness**: 90% of time-sensitive queries return current information
- **Query Success Rate**: >95% of search queries return usable results
- **False Positive Rate**: <10% irrelevant results in top 5 search results

### Integration Metrics

- **Provider Failover**: <500ms to switch between search providers
- **Rate Limit Handling**: Zero query failures due to rate limiting
- **Content Processing**: <200ms overhead for content extraction and filtering
- **Cache Effectiveness**: 60% reduction in API calls through intelligent caching

## Implementation Strategy

### Phase 1A: Core Search Integration (Week 1-2)

- Implement Tavily integration with basic query routing
- Add simple content extraction and relevance filtering
- Test with common coding queries and validate latency targets

### Phase 1B: Multi-Provider Setup (Week 3-4)

- Add Exa integration for specialized technical queries
- Implement rate limiting and fallback mechanisms
- Add intelligent caching with content-type-aware TTL

### Phase 1C: Production Optimization (Week 5-6)

- Optimize query routing and provider selection algorithms
- Add comprehensive monitoring and cost tracking
- Load testing and capacity planning for production usage

### Phase 2 Extensions

- ZenRows integration for complex site scraping
- Advanced content analysis and summarization
- Federated search result sharing across instances
