# ADR-003: Tool Integration Protocol

**Status**: Accepted

**Context**: CodeForge AI needs low-latency tool integration for Phase 1 core functionality, with extensibility for Phase 2 advanced tools like vision SDK. Must balance direct performance with customization needs while maintaining minimal overhead for frequently used operations. Core database operations require <10ms latency while custom tools can tolerate higher latency.

**Decision**: Direct SDKs for core tools (qdrant-client v1.15.0+, neo4j v5.28.1+, redis v6.0.0+) in Phase 1, MCP for custom integrations, extend with openai SDK v1.97.0+ for Phase 2 vision.

**Consequences**:

- Positive: Low latency for critical operations, easy Phase 2 extensions, proven SDK reliability, optimal performance for core database and model operations, consistent error handling
- Negative: Mixed integration patterns increase complexity, need to maintain both SDK and MCP integrations, different debugging approaches per tool type

## Architecture Overview

### Dual Integration Strategy

- **Core SDK Tools**: Direct integration for database operations and critical services
- **MCP Tools**: Model Context Protocol for external services and custom integrations
- **Performance Tiering**: Different latency budgets based on tool criticality
- **Unified Interface**: Consistent tool interface despite different underlying protocols

### Core Tool Requirements

- **Qdrant**: Vector operations with <50ms latency budget
- **Neo4j**: Graph queries with <100ms latency budget  
- **Redis**: Cache operations with <10ms latency budget
- **OpenRouter**: Model routing with <2000ms acceptable latency
- **Web Search**: Tavily/Exa with <2000ms latency tolerance

### Tool Management Framework

- **Connection Pooling**: Optimized connection management per tool type
- **Health Monitoring**: Continuous tool availability checking
- **Automatic Retry**: Intelligent retry with exponential backoff
- **Circuit Breakers**: Prevent cascade failures from tool unavailability
- **Performance Tracking**: Real-time latency and success rate monitoring

## Phase 2 Extensions

### Vision Tool Integration

- **OpenAI SDK v1.97.0+**: GPT-4V for image analysis and UI understanding
- **CLIP Embeddings**: Multi-modal search capabilities
- **Advanced Scraping**: ZenRows for complex web content extraction
- **Performance Targets**: <5s for complex vision tasks, <30s timeout

### MCP Ecosystem Expansion

- **Custom Tools**: Domain-specific integrations via MCP protocol
- **Third-Party Services**: API integrations through standardized MCP interface
- **Tool Discovery**: Dynamic tool registration and capability negotiation
- **Version Management**: Backward compatibility and graceful upgrades

### Implementation Architecture

```pseudocode
ToolIntegrationManager {
  coreSDKTools: Map<ToolName, SDKInstance>
  mcpTools: Map<ToolName, MCPClient>
  performanceMetrics: ToolMetrics
  
  executeOperation(toolName, operation, params) -> Result {
    tool = getToolInstance(toolName)
    
    startTime = now()
    result = tool.execute(operation, params)
    executionTime = now() - startTime
    
    trackPerformance(toolName, executionTime)
    validateLatencyBudget(toolName, executionTime)
    
    return result
  }
}

PerformanceMonitoring {
  latencyBudgets: Map<ToolName, Milliseconds>
  healthChecks: ScheduledChecks
  circuitBreakers: FailureProtection
  
  // Continuous monitoring and alerting
}
```

## Success Criteria

### Performance Targets

- **Core SDK Latency**: Qdrant <50ms, Neo4j <100ms, Redis <10ms
- **Tool Availability**: >99% uptime for core tools, >95% for MCP tools
- **Connection Efficiency**: Connection pool utilization >70%, <20 concurrent connections per tool
- **Error Recovery**: <5% failed operations, automatic retry success rate >90%
- **Throughput**: 1000+ tool operations per minute sustained load

### Integration Quality

- **SDK Version Alignment**: All core SDKs match pyproject.toml versions exactly
- **MCP Compatibility**: Support for standard MCP protocol with fallback mechanisms
- **Extension Readiness**: Phase 2 vision tools integrate with <1 day development time
- **Unified Interface**: Consistent API regardless of underlying tool protocol

### Operational Metrics

- **Health Check Success**: >98% successful health checks across all tools
- **Resource Utilization**: <50% CPU overhead for tool management layer
- **Memory Efficiency**: <200MB total overhead for all tool connections
- **Monitoring Coverage**: 100% of tool operations tracked with metrics

## Implementation Strategy

### Phase 1A: Core SDK Integration (Week 1-2)

- Implement direct SDK integrations for Qdrant, Neo4j, Redis
- Establish connection pooling and performance monitoring
- Validate latency targets with load testing

### Phase 1B: MCP Tool Integration (Week 3-4)

- Add MCP adapter for web search tools (Tavily, Exa)
- Implement health checking and retry mechanisms
- Test hybrid SDK + MCP workflows under realistic load

### Phase 1C: Production Hardening (Week 5-6)

- Add comprehensive error handling and circuit breakers
- Implement automated performance reporting and alerting
- Prepare Phase 2 extension points and vision tool architecture
