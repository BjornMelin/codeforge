# ADR-001: Multi-Agent Framework Selection

**Status**: Accepted  

**Date**: 2025-07-20  

**Deciders**: CodeForge AI Team  

## Context

Need flexible orchestration for hierarchical/debate/multi-model in Phase 1, extendable to Phase 2 multi-modal/federated. The framework must support complex workflows while maintaining simplicity for autonomous flows across phases.

## Problem Statement

Balance power/simplicity for autonomous flows across phases. The system needs to handle:

- Hierarchical agent coordination

- Debate mechanisms between agents

- Multi-model routing and coordination

- Extensibility for Phase 2 features (multi-modal, federated learning)

## Decision

**LangGraph v0.3** for graph-based/stateful orchestration in Phase 1; extend subgraphs for Phase 2 features.

## Alternatives Considered

| Framework | Pros | Cons | Score |
|-----------|------|------|-------|
| **LangGraph v0.3** | Graph-based, stateful, subgraph support, excellent for debate/hierarchy | Learning curve, newer framework | **8.9** |
| CrewAI | Simple API, good documentation, easy setup | Less flexible for complex workflows, limited state management | 7.8 |
| AutoGen | Strong debate capabilities, Microsoft backing | Complex setup, overkill for MVP | 7.4 |
| Custom Framework | Full control, optimized for use case | High development time, maintenance burden | 6.5 |

## Rationale

- **High leverage/value (8.9)**: Supports subgraphs/debate in Phase 1

- **Easy extension**: Natural path to multi-modal workflows in Phase 2

- **State management**: Built-in StateGraph for shared context

- **Community**: Growing ecosystem with LangSmith integration

## Consequences

### Positive

- Easy integration with LLM providers

- Built-in state persistence and checkpointing

- Use LangSmith for tracing in both phases

- Natural support for complex agent workflows

### Negative

- Learning curve for new framework

- Potential breaking changes in early versions

- Dependency on LangChain ecosystem

### Neutral

- Framework choice affects all subsequent architectural decisions

## Implementation Notes

```python
from langgraph.graph import StateGraph, add_messages
from typing import TypedDict, Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]
    task_queue: deque
    private: dict

workflow = StateGraph(State)
workflow.add_node('orchestrator', orchestrator_node)
workflow.add_node('debate', debate_subgraph)
graph = workflow.compile(checkpointer=MemorySaver())
```

## Related Decisions

- ADR-004: Orchestration and Task Management

- ADR-009: Debate Agents

- ADR-005: Caching and Shared Context Layer

## Monitoring

- Track workflow execution times

- Monitor state management overhead

- Measure debate effectiveness metrics

- LangSmith traces for debugging
