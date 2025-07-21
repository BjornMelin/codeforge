# ADR-005: Caching and Shared Context Layer

**Status**: Accepted

**Context**: CodeForge AI needs real-time shared state management across agents in Phase 1, extensible to Phase 2 federated privacy requirements. Critical for preventing hallucination and enabling effective collaboration with <1ms access times for frequent operations. Must maintain consistency across 3-agent debates and scale to 5-agent configurations while preventing information drift and ensuring all agents work from the same factual base.

**Decision**: LangGraph v0.5.3+ StateGraph with hierarchical memory architecture - short-term shared messages in-memory, long-term checkpointers to SQLite/DB in Phase 1, extending with federated local states in Phase 2.

**Consequences**:

- Positive: Automatic state synchronization across agents, built-in anti-hallucination mechanisms through shared truth, hierarchical memory supports different access patterns, natural extension to federated privacy models, prevents context drift between agents
- Negative: Framework dependency on LangGraph, need for careful state size management to avoid memory bloat, potential single point of failure for shared state

## Architecture Overview

### Hierarchical Memory Design

- **Short-term Layer**: Active session data, current task context, agent coordination state (in-memory for <1ms access)
- **Long-term Layer**: Conversation history, learned patterns, persistent facts (SQLite/DB for durability)
- **Cache Layer**: Frequently accessed data with TTL management (5-minute default)

### Anti-Hallucination Strategy

- **Shared Truth Repository**: Single source of truth for facts across all agents
- **Consistency Validation**: Real-time checking of agent outputs against shared context
- **Automatic Correction**: System-generated corrections for inconsistent agent responses
- **Fact Verification**: Cross-reference agent claims with established knowledge base

### State Management Requirements

- **Consistency**: 100% state consistency across agents in same session
- **Performance**: <1ms access for short-term, <10ms for long-term memory
- **Scalability**: Support 10+ concurrent agents in Phase 1, 50+ in Phase 2
- **Durability**: Zero data loss for important decisions and learned patterns
- **Privacy**: Hierarchical access controls for federated Phase 2 scenarios

## Phase 2 Extensions

### Federated Context Management

- **Privacy Levels**: Strict (patterns only), Moderate (anonymized insights), Open (full sharing)
- **Local State Isolation**: Each federated node maintains private context
- **Selective Synchronization**: Privacy-filtered context sharing across nodes
- **Aggregate Learning**: Federated insights without exposing private data

### Implementation Architecture

```pseudocode
StateManager {
  shortTermMemory: InMemoryCache<1ms
  longTermMemory: PersistentStorage<10ms  
  factRepository: SharedTruthBase
  
  getContext(agentId, contextType) -> Context
  updateContext(agentId, updates) -> bool
  validateOutput(agentId, output) -> ValidationResult
}

AntiHallucinationLayer {
  verifyFacts(output) -> FactCheckResult
  checkConsistency(output, sharedContext) -> ConsistencyResult
  generateCorrections(issues) -> CorrectionList
}
```

## Success Criteria

### Performance Targets

- **Context Access**: <1ms short-term, <10ms long-term memory access
- **State Synchronization**: <100ms for cross-agent updates
- **Memory Efficiency**: <50MB total state size per active session
- **Cache Hit Rate**: >90% for frequently accessed context
- **Anti-Hallucination Accuracy**: <2% false positive rate

### Reliability Metrics

- **State Consistency**: 100% consistency across agents in same session
- **Data Persistence**: Zero data loss for long-term memory
- **Recovery Time**: <5s to restore state after system restart
- **Concurrent Access**: Support 10+ agents simultaneously without conflicts
- **Truth Validation**: >95% accuracy in fact checking and correction

### Quality Metrics

- **Hallucination Reduction**: 40% fewer inconsistent responses vs baseline
- **Context Relevance**: >85% relevance score for retrieved context
- **Agent Coordination**: 30% improvement in multi-agent task completion
- **Memory Utilization**: <500MB peak memory usage for 10 concurrent sessions

## Implementation Strategy

### Phase 1A: Core Architecture (Week 1-2)

- Implement LangGraph StateGraph with hierarchical memory
- Basic shared context management and caching
- Simple anti-hallucination with fact checking

### Phase 1B: Advanced Features (Week 3-4)

- Sophisticated consistency validation
- Automatic correction mechanisms
- Performance optimization and monitoring

### Phase 1C: Production Hardening (Week 5-6)

- Comprehensive error handling and recovery
- Load testing with target performance metrics
- Documentation and federated extension points
