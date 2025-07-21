# ADR-007: Web Search Integration

**Status**: Accepted  

**Date**: 2025-07-20  

**Deciders**: CodeForge AI Team  

## Context

Handle misses and current events in RAG for Phase 1, extend to Phase 2 advanced scraping. The system needs reliable web search capabilities to complement local knowledge with real-time information and handle queries beyond the local knowledge base.

## Problem Statement

Complement local RAG with web search for knowledge gaps across phases. Requirements include:

- Real-time information access for current events

- Cost-effective API usage for frequent queries

- Backup options for API reliability

- Deep scraping capabilities for Phase 2

- Integration with GraphRAG+ pipeline

## Decision

**Tavily primary + Exa secondary**, agentic trigger in Phase 1; extend to Phase 2 with ZenRows alternative for deeper scraping.

## Alternatives Considered

| Solution | Pros | Cons | Score |
|----------|------|------|-------|
| **Tavily + Exa Hybrid** | Cost-effective, reliable, complementary strengths | Multiple API dependencies | **8.6** |
| Firecrawl only | High quality scraping, comprehensive | Expensive for frequent use | 7.2 |
| Pure local search | No API costs, full control | Slower, incomplete coverage | 7.0 |
| Google Search API | High quality, comprehensive | Expensive, rate limits | 7.8 |

## Rationale

- **Cheap/accurate (8.6)**: Phase 1 cost-effectiveness with quality results

- **Phase 1 complement**: Fills gaps in local knowledge effectively

- **Phase 2 depth**: ZenRows adds advanced scraping capabilities

- **Agentic integration**: Smart triggering reduces unnecessary API calls

## Consequences

### Positive

- Comprehensive information coverage beyond local knowledge

- Cost-effective through intelligent API selection

- Reliable fallback options prevent single points of failure

- Strong foundation for Phase 2 advanced scraping

### Negative

- External API dependencies affect reliability

- API rate limits may constrain usage patterns

- Multiple integrations increase system complexity

### Neutral

- API keys required in Phase 1

- Toggle ZenRows integration in Phase 2

## Implementation Notes

### Web Search Orchestrator
```python
from typing import List, Dict, Optional, Union
import asyncio
from dataclasses import dataclass

@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    relevance_score: float
    source: str  # 'tavily', 'exa', 'zenrows'
    timestamp: float

class WebSearchOrchestrator:
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
        self.exa_client = ExaClient(api_key=os.getenv('EXA_API_KEY'))
        self.zenrows_client = None  # Phase 2
        self.usage_tracker = APIUsageTracker()
        self.cache = SearchCache(ttl=3600)  # 1 hour cache
    
    async def search(self, query: str, search_type: str = 'hybrid') -> List[SearchResult]:
        """Main search orchestration with intelligent routing"""
        
        # Check cache first
        cached_results = self.cache.get(query)
        if cached_results:
            return cached_results
        
        # Determine search strategy
        if search_type == 'hybrid':
            results = await self._hybrid_search(query)
        elif search_type == 'deep':
            results = await self._deep_search(query)  # Phase 2
        else:
            results = await self._primary_search(query)
        
        # Cache and return
        self.cache.set(query, results)
        return results
    
    async def _hybrid_search(self, query: str) -> List[SearchResult]:
        """Parallel search across multiple providers"""
        
        tasks = []
        
        # Always try Tavily (primary)
        tasks.append(self._tavily_search(query))
        
        # Use Exa for specific query types
        if self._should_use_exa(query):
            tasks.append(self._exa_search(query))
        
        # Execute searches in parallel
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge and rank results
        all_results = []
        for results in search_results:
            if isinstance(results, list):
                all_results.extend(results)
        
        return self._merge_and_rank(all_results, query)
```

### Provider-Specific Implementations
```python
class TavilySearcher:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)
        self.rate_limiter = RateLimiter(calls_per_minute=100)
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Tavily search with rate limiting and error handling"""
        
        await self.rate_limiter.acquire()
        
        try:
            response = await self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=["github.com", "stackoverflow.com", "docs.python.org"],
                exclude_domains=["pinterest.com", "facebook.com"]
            )
            
            return [
                SearchResult(
                    title=result['title'],
                    url=result['url'],
                    content=result['content'],
                    relevance_score=result.get('score', 0.5),
                    source='tavily',
                    timestamp=time.time()
                )
                for result in response.get('results', [])
            ]
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

class ExaSearcher:
    def __init__(self, api_key: str):
        self.client = ExaClient(api_key=api_key)
        self.rate_limiter = RateLimiter(calls_per_minute=60)
    
    async def search(self, query: str, search_type: str = 'neural') -> List[SearchResult]:
        """Exa search optimized for technical content"""
        
        await self.rate_limiter.acquire()
        
        try:
            if search_type == 'neural':
                response = await self.client.search_and_contents(
                    query=query,
                    num_results=8,
                    text={"max_characters": 2000},
                    highlights={"num_sentences": 3}
                )
            else:
                response = await self.client.find_similar_and_contents(
                    url=query,  # For similar document finding
                    num_results=5
                )
            
            return [
                SearchResult(
                    title=result.title,
                    url=result.url,
                    content=result.text,
                    relevance_score=result.score,
                    source='exa',
                    timestamp=time.time()
                )
                for result in response.results
            ]
            
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return []
```

### Agentic Search Triggering
```python
class AgenticSearchTrigger:
    def __init__(self, local_rag_system):
        self.local_rag = local_rag_system
        self.confidence_threshold = 0.7
        self.freshness_threshold = 86400  # 24 hours
    
    async def should_trigger_web_search(self, query: str, local_results: List[Dict]) -> bool:
        """Intelligent decision on when to use web search"""
        
        # Check local result confidence
        if not local_results:
            return True
        
        max_confidence = max(result.get('confidence', 0) for result in local_results)
        if max_confidence < self.confidence_threshold:
            return True
        
        # Check for temporal queries
        if self._is_temporal_query(query):
            return True
        
        # Check for unknown technologies/libraries
        if self._contains_unknown_tech(query, local_results):
            return True
        
        # Check for current events
        if self._is_current_event_query(query):
            return True
        
        return False
    
    def _is_temporal_query(self, query: str) -> bool:
        """Detect queries asking for current/recent information"""
        temporal_indicators = [
            'latest', 'newest', 'current', 'recent', '2024', '2025',
            'now', 'today', 'this year', 'updated', 'new'
        ]
        return any(indicator in query.lower() for indicator in temporal_indicators)
    
    def _contains_unknown_tech(self, query: str, local_results: List[Dict]) -> bool:
        """Check if query mentions technologies not in local knowledge"""
        # Extract technology mentions from query
        tech_mentions = self._extract_tech_mentions(query)
        
        # Check if any are missing from local results
        for tech in tech_mentions:
            if not any(tech.lower() in result.get('content', '').lower() 
                      for result in local_results):
                return True
        
        return False
```

### Results Fusion and Ranking
```python
class SearchResultsFusion:
    def __init__(self):
        self.relevance_weights = {
            'tavily': 0.4,
            'exa': 0.35,
            'zenrows': 0.25  # Phase 2
        }
        self.domain_boost = {
            'github.com': 1.2,
            'stackoverflow.com': 1.15,
            'docs.python.org': 1.1,
            'arxiv.org': 1.05
        }
    
    def merge_and_rank(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Intelligent fusion of results from multiple sources"""
        
        # Remove duplicates based on URL similarity
        unique_results = self._deduplicate_results(results)
        
        # Calculate combined relevance scores
        for result in unique_results:
            result.relevance_score = self._calculate_combined_score(result, query)
        
        # Sort by relevance and return top results
        sorted_results = sorted(
            unique_results, 
            key=lambda r: r.relevance_score, 
            reverse=True
        )
        
        return sorted_results[:20]  # Top 20 results
    
    def _calculate_combined_score(self, result: SearchResult, query: str) -> float:
        """Calculate combined relevance score"""
        base_score = result.relevance_score
        
        # Apply source weight
        weighted_score = base_score * self.relevance_weights.get(result.source, 1.0)
        
        # Apply domain boost
        domain = self._extract_domain(result.url)
        domain_multiplier = self.domain_boost.get(domain, 1.0)
        
        # Apply content relevance boost
        content_boost = self._calculate_content_relevance(result.content, query)
        
        # Apply recency boost for time-sensitive queries
        recency_boost = self._calculate_recency_boost(result.timestamp, query)
        
        final_score = weighted_score * domain_multiplier * content_boost * recency_boost
        
        return min(final_score, 1.0)  # Cap at 1.0
```

## Phase 2 Extensions

### Advanced Scraping with ZenRows
```python
class ZenRowsDeepScraper:  # Phase 2
    def __init__(self, api_key: str):
        self.client = ZenRowsClient(api_key=api_key)
        self.rate_limiter = RateLimiter(calls_per_minute=30)
    
    async def deep_scrape(self, urls: List[str]) -> List[SearchResult]:
        """Deep scraping for comprehensive content extraction"""
        
        results = []
        for url in urls:
            await self.rate_limiter.acquire()
            
            try:
                response = await self.client.scrape(
                    url=url,
                    params={
                        'js_render': True,
                        'wait': 3000,
                        'block_resources': 'image,media',
                        'custom_headers': True
                    }
                )
                
                content = self._extract_main_content(response.content)
                
                results.append(SearchResult(
                    title=self._extract_title(response.content),
                    url=url,
                    content=content,
                    relevance_score=0.8,  # High confidence for deep scrape
                    source='zenrows',
                    timestamp=time.time()
                ))
                
            except Exception as e:
                logger.error(f"ZenRows scraping failed for {url}: {e}")
        
        return results
```

## Cost Management

### API Usage Optimization
```python
class APIUsageTracker:
    def __init__(self):
        self.usage_limits = {
            'tavily': {'daily': 1000, 'current': 0},
            'exa': {'daily': 500, 'current': 0},
            'zenrows': {'daily': 200, 'current': 0}  # Phase 2
        }
        self.cost_per_call = {
            'tavily': 0.001,  # $0.001 per call
            'exa': 0.002,     # $0.002 per call
            'zenrows': 0.01   # $0.01 per call (Phase 2)
        }
    
    def can_make_call(self, service: str) -> bool:
        """Check if we can make another API call"""
        current = self.usage_limits[service]['current']
        limit = self.usage_limits[service]['daily']
        return current < limit
    
    def record_call(self, service: str) -> float:
        """Record API call and return cost"""
        self.usage_limits[service]['current'] += 1
        return self.cost_per_call[service]
    
    def get_daily_cost(self) -> float:
        """Calculate total daily API costs"""
        total_cost = 0
        for service, limits in self.usage_limits.items():
            calls_made = limits['current']
            cost_per_call = self.cost_per_call[service]
            total_cost += calls_made * cost_per_call
        return total_cost
```

## Performance Targets

| Metric | Phase 1 Target | Phase 2 Target |
|--------|----------------|----------------|
| Search Latency | <2s | <3s (with deep scrape) |
| Cache Hit Rate | >60% | >70% |
| Daily API Cost | <$10 | <$25 |
| Result Relevance | >0.8 | >0.85 |

## Related Decisions

- ADR-006: SOTA GraphRAG Implementation

- ADR-008: Multi-Model Routing

## Monitoring

- API usage and cost tracking

- Search result quality metrics

- Cache performance and hit rates

- Provider availability and error rates

- Query pattern analysis for optimization
