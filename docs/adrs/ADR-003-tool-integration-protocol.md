# ADR-003: Tool Integration Protocol

**Status**: Accepted  

**Date**: 2025-07-20  

**Deciders**: CodeForge AI Team  

## Context

Latency/extensibility for tools in Phase 1, extend to Phase 2 advanced (e.g., vision SDK). The system needs to balance direct performance with custom tool integration across both phases while maintaining low latency and high extensibility.

## Problem Statement

Balance direct performance with custom integration in phases. Requirements include:

- Low-latency access to core tools

- Extensibility for custom tool development

- Integration with existing codebase tools

- Future support for advanced tools (vision, federated)

- Minimal overhead for frequently used operations

## Decision

**Direct SDKs** for core in Phase 1 (e.g., qdrant-client); **MCP for custom**; extend to Phase 2 with openai SDK for vision.

## Alternatives Considered

| Approach | Pros | Cons | Score |
|----------|------|------|-------|
| **Direct SDKs + MCP** | Low latency for core, extensible for custom | Mixed integration patterns | **8.5** |
| All MCP | Consistent interface, easy extension | Latency overhead for all operations | 7.0 |
| All Direct SDKs | Maximum performance | Limited extensibility, custom tool burden | 7.8 |
| Custom Wrapper Layer | Unified interface, controllable | Development overhead, maintenance burden | 7.2 |

## Rationale

- **Low load/high performance (8.5)**: Direct access for frequently used tools

- **Easy Phase 2 addition**: Natural extension points for vision and federated tools

- **Best of both worlds**: Performance where needed, flexibility where useful

- **Codebase integration**: Leverage existing tool wrappers

## Consequences

### Positive

- Optimal performance for core database and model operations

- Easy integration of specialized tools through MCP

- Clear separation between performance-critical and extensible tools

- Future-proof for Phase 2 advanced capabilities

### Negative

- Mixed integration patterns increase complexity

- Need to maintain both SDK and MCP integration paths

- Potential for inconsistent error handling across tools

### Neutral

- Wrappers needed for codebase integration in Phase 1

- Toggle vision capabilities in Phase 2

## Implementation Notes

### Core Tools (Direct SDK)
```python

# High-frequency, performance-critical tools
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import redis
from openrouter import OpenRouter

class CoreTools:
    def __init__(self):
        self.qdrant = QdrantClient("localhost", port=6333)
        self.neo4j = GraphDatabase.driver("bolt://localhost:7687")
        self.redis = redis.Redis(host='localhost', port=6379)
        self.router = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
    
    async def vector_search(self, query_vector, limit=10):
        return await self.qdrant.search(
            collection_name="knowledge",
            query_vector=query_vector,
            limit=limit
        )
```

### Custom Tools (MCP)
```python

# Extensible tools through MCP
from mcp_client import MCPClient

class CustomTools:
    def __init__(self):
        self.mcp_client = MCPClient()
    
    async def tree_sitter_analysis(self, code, language):
        return await self.mcp_client.call_tool(
            "tree_sitter_analyze",
            {"code": code, "language": language}
        )
    
    async def git_operations(self, repo_path, operation):
        return await self.mcp_client.call_tool(
            "git_operation",
            {"path": repo_path, "operation": operation}
        )
```

### Tool Registry
```python
class ToolRegistry:
    def __init__(self):
        self.core_tools = CoreTools()
        self.custom_tools = CustomTools()
        self.phase2_tools = Phase2Tools()  # Vision, federated
    
    async def route_tool_call(self, tool_name, **kwargs):
        if tool_name in self.CORE_TOOLS:
            return await getattr(self.core_tools, tool_name)(**kwargs)
        elif tool_name in self.CUSTOM_TOOLS:
            return await self.custom_tools.call_tool(tool_name, kwargs)
        elif tool_name in self.PHASE2_TOOLS:
            return await getattr(self.phase2_tools, tool_name)(**kwargs)
        else:
            raise ToolNotFoundError(f"Tool {tool_name} not registered")
```

## Tool Categories

### Phase 1 Core Tools (Direct SDK)

- **Database Operations**: Qdrant, Neo4j, Redis

- **Model Routing**: OpenRouter API

- **Basic File Operations**: Standard library

- **HTTP Requests**: httpx for web search APIs

### Phase 1 Custom Tools (MCP)

- **Code Analysis**: tree-sitter parsing

- **Git Operations**: libgit2 wrappers

- **Search Tools**: Tavily, Exa integrations

- **Custom Analyzers**: Domain-specific tools

### Phase 2 Advanced Tools (Direct SDK)

- **Vision Processing**: OpenAI SDK for CLIP

- **Federated Learning**: Flower framework

- **Advanced Scraping**: ZenRows for deep web access

- **Kubernetes**: K8s client for orchestration

## Performance Expectations

| Tool Category | Latency Target | Throughput Target |
|---------------|----------------|-------------------|
| Core SDK | <10ms | 1000+ req/s |
| Custom MCP | <50ms | 100+ req/s |
| Phase 2 Vision | <200ms | 50+ req/s |
| Phase 2 Federated | <500ms | 10+ req/s |

## Error Handling Strategy

```python
class ToolError(Exception):
    def __init__(self, tool_name, operation, error_details):
        self.tool_name = tool_name
        self.operation = operation
        self.error_details = error_details
        super().__init__(f"Tool {tool_name} failed on {operation}: {error_details}")

async def safe_tool_call(tool_func, *args, **kwargs):
    try:
        return await tool_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Tool call failed: {tool_func.__name__}", exc_info=True)
        raise ToolError(tool_func.__name__, "execution", str(e))
```

## Related Decisions

- ADR-002: Database and Memory System

- ADR-006: SOTA GraphRAG Implementation

- ADR-011: Multi-Modal Support (Phase 2)

## Monitoring

- Tool execution latencies by category

- Error rates and failure patterns

- MCP vs Direct SDK performance comparison

- Resource utilization per tool type

- Phase 2 tool adoption metrics
