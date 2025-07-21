# Product Requirements Document (PRD) for CodeForge AI

## Executive Summary

CodeForge AI is a future-proof, low-maintenance multi-agent system automating software development from idea/PRD generation to code/tests/deploy/PRs. Phase 1 (MVP) focuses on core autonomy with SOTA GraphRAG+, multi-model routing, debate agents, hybrid task management, and shared state. Phase 2 extends with multi-modal (vision for UI), advanced embeddings (SPLADE hybrid), extended debate (5-agent toggle), and optimizations like federated learning basics (if scaled). Built for solo devs with <$200/month costs and local/Docker deploy, it migrates/enhances Claude Code assets. Success: Phase 1 (95% task completion, <100ms latency); Phase 2 (multi-modal acc +20%, federated privacy).

**Vision**: Virtual dev team forging code autonomously across phases.

**Target Users**: Solo/small teams; Phase 2 adds multi-modal for UI/web devs.

**Goals**: Phase 1: 80% time reduction; Phase 2: 90% with advanced features.

**Metrics**: Phase 1: <10% hallucination; Phase 2: +20% multi-modal acc.

## Problem Statement

Phase 1 solves core issues (context loss, fragmented tools). Phase 2 addresses advanced needs: Multi-modal input (e.g., UI images for testing), deeper lexical retrieval (SPLADE for sparse), scalable debate (5 agents for complex), and privacy in collaboration (federated basics).

## Key Features

### Phase 1 Features (MVP)

1. **Autonomous Orchestration**: LangGraph graphs for workflows; hierarchical agents with 3-agent debate.
2. **SOTA Retrieval**: GraphRAG+ with agentic hybrid RAG, content-varied embeddings (384/256 code, 768/512 PDFs), Tavily/Exa web.
3. **Multi-Model Routing**: OpenRouter dynamic (Grok 4 ~40%, Claude 4 ~30%, Kimi K2 ~20%, Gemini flash ~10%, o3 <5%).
4. **Task Management**: Hybrid in-memory deque + Redis Pub/Sub/checkpointers.
5. **Shared State**: StateGraph hierarchical (short-term shared messages, long-term persistent).
6. **Enhanced Tools**: Wrapped atomics (tree-sitter, libgit2).
7. **Autonomy Flows**: One-shot gen/feature PRs with debate/review.

### Phase 2 Features (Optimizations/Extensions)

1. **Multi-Modal Support**: Vision integration (openai SDK for UI testing/images in RAG; e.g., embed images with CLIP models for 20% acc boost in web dev).
2. **Advanced Embeddings**: SPLADE for sparse lexical (hybrid with BGE-M3 dense for +15% precision in code/papers); dynamic fusion.
3. **Extended Debate**: Toggle to 5 agents (e.g., add advocate/critic/refiner for toughest tasks, +10% gains in complex reasoning).
4. **Federated Basics**: Simple federated learning (e.g., Flower lib for privacy in multi-agent training on local data; basic model aggregation for personalized agents).
5. **Enhanced Optimizations**: Adaptive quantization (e.g., dynamic int4/8 based on content), extended web (ZenRows alt for scraping if needed), and scalability (Kubernetes opt-in for >100 agents).

## Functional Requirements

### Phase 1 Functional Requirements

| ID | Feature | Description | Priority | Acceptance Criteria |
|----|---------|-------------|----------|---------------------|
| FR-01 | Orchestration | LangGraph graphs with subgraphs for hierarchy/debate. | High | Executes one-shot/feature flows; debate boosts acc 30%. |
| FR-02 | Retrieval | Agentic hybrid GraphRAG+ with web (Tavily/Exa) for misses; vary embeddings/chunks by type. | High | Retrieves with 30-40% acc; auto-updates KG. |
| FR-03 | Model Routing | Dynamic classification/intensity-based selection via OpenRouter. | High | Routes to specialized models; saves 60% costs. |
| FR-04 | Task Mgmt | Hybrid in-memory queue + Redis sync/Pub/Sub. | High | Low-latency pull/update; no overlaps. |
| FR-05 | Shared State | Hierarchical StateGraph with shared/private memory. | High | Anti-hallucination collaboration; <1ms access. |
| FR-06 | Tools/Scripts | Wrapped atomics with enhancements (e.g., tree-sitter analysis). | Medium | Reuses codebase; integrates with RAG. |
| FR-07 | Autonomy | One-shot gen/feature PRs with debate/review/validation. | High | Full cycle without intervention; ethical checks. |

### Phase 2 Functional Requirements

| ID | Feature | Description | Priority | Acceptance Criteria |
|----|---------|-------------|----------|---------------------|
| FR-08 | Multi-Modal | Vision SDK for UI/image in RAG (e.g., CLIP embeddings for +20% acc in web/UI tasks). | Medium | Processes images in retrieval/workflows; validates UI tests. |
| FR-09 | Advanced Embeddings | SPLADE sparse hybrid with BGE-M3 (+15% precision); dynamic fusion by content. | Medium | Improves lexical search for code/papers; toggles for perf. |
| FR-10 | Extended Debate | 5-agent toggle (add advocate/critic/refiner; +10% gains in complex). | Medium | Scales debate for toughest tasks; iterates 3 rounds if needed. |
| FR-11 | Federated Basics | Flower lib for local model aggregation (privacy in agent training). | Low | Aggregates models without central data; basic personalization. |
| FR-12 | Enhanced Opts | Adaptive quantization (int4/8 dynamic); ZenRows web alt; K8s scalability. | Low | Reduces memory 50%; supports >100 agents; toggles for advanced. |

## Non-Functional Requirements

### Phase 1 Non-Functional Requirements

| ID | Category | Requirement | Metric | Validation |
|----|----------|-------------|--------|------------|
| NFR-01 | Performance | Latency/acc | <100ms, +30% acc | Code_execution/tests. |
| NFR-02 | Cost | Monthly | <$200 | OpenRouter tracking. |
| NFR-03 | Maintainability | Upgrades | Lib-first/toggles | Minimal custom. |
| NFR-04 | Scalability | Agents | 100+ | Docker Compose. |
| NFR-05 | Reliability | Uptime | 99% | Tenacity/retries. |

### Phase 2 Non-Functional Requirements

| ID | Category | Requirement | Metric | Validation |
|----|----------|-------------|--------|------------|
| NFR-06 | Performance | Multi-modal acc | +20% | Vision benchmarks. |
| NFR-07 | Cost | Advanced ops | <$50 add | Usage tracking. |
| NFR-08 | Maintainability | Toggles | Zero-impact upgrades | Feature flags. |
| NFR-09 | Scalability | Agents | 500+ | K8s tests. |
| NFR-10 | Reliability | Federated | 99.5% | Aggregation sims. |

## Scope & Prioritization

- **Phase 1 (MVP)**: Core autonomy (orchestration, GraphRAG+, routing, debate, task/state)3-5 days.

- **Phase 2**: Extensions (multi-modal, advanced embeddings/debate, federated basics)1 week post-MVP; prioritize by value (multi-modal first for UI/web).

- **Out of Scope**: Full federated (beyond basics), enterprise monitoring.

## Dependencies & Integrations

### Phase 1 Dependencies

- **Frameworks**: LangGraph v0.3

- **Models**: OpenRouter (Grok 4/Claude Sonnet 4/Kimi K2/Gemini 2.5-flash/o3)

- **DB**: Neo4j/Qdrant/Redis

- **Embeddings**: sentence-transformers v3.0 (BAAI/bge-large-en-v1.5/BGE-M3)

- **Tools**: Tavily/Exa, tree-sitter

- **Libs**: Python 3.12/3.13, httpx/asyncio, tenacity, ruff, pytest, Poetry

### Phase 2 Dependencies

- **Multi-Modal**: openai SDK for vision (CLIP models)

- **Advanced Embeddings**: SPLADE via FastEmbed

- **Extended Debate**: No new libs (LangGraph toggle)

- **Federated**: Flower lib for aggregation

- **Scalability**: Kubernetes (opt-in)

- **Libs**: Add ZenRows for web alt if needed

## Risks & Mitigations

### Phase 1 Risks

- **Routing latency**: Simple if-else/LangSmith monitoring

- **State bloat**: Message limits/checkpointers

- **Costs overrun**: Route 70% to cheap models

### Phase 2 Risks

- **Multi-Modal latency**: Toggle for non-UI tasks

- **Federated privacy**: Basic aggregation only; test locally

- **Scalability overload**: K8s opt-in, start with Docker

## Success Metrics

### Phase 1 Success Metrics

- Task completion: 95%

- Latency: <100ms

- Accuracy improvement: +30-40%

- Cost: <$200/month

- Hallucination rate: <10%

### Phase 2 Success Metrics

- Multi-modal accuracy: +20%

- Advanced embeddings precision: +15%

- Extended debate gains: +10%

- Federated privacy: Basic aggregation working

- Scalability: Support 500+ agents

## Timeline

### Phase 1: MVP (3-5 Days)

- **Day 1**: Poetry deps, tools.py wrappers

- **Day 2**: Hybrid DB setup, GraphRAG+ implementation

- **Day 3**: StateGraph, hybrid task management

- **Day 4**: Multi-model routing, 3-agent debate

- **Day 5**: Full autonomy flows, testing, Docker Compose

### Phase 2: Extensions (1 Week Post-MVP)

- **Day 1**: Multi-modal integration

- **Day 2**: Advanced embeddings (SPLADE)

- **Day 3**: Extended debate (5 agents)

- **Day 4**: Federated basics

- **Day 5**: Enhanced scalability, full testing

## Implementation Plans

### Phase 1: MVP Implementation

**Overview**: Build core autonomy with LangGraph, hybrid DB, routing, debate, task/state.

**Detailed Implementation**:

- **Day 1**: Poetry deps (LangGraph, sentence-transformers, openrouter, neo4j-driver, qdrant-client, redis-py, httpx, tenacity, ruff, pytest); tools.py wrappers for atomics (e.g., atomic-analysis/files/git/search with tree-sitter/libgit2/GraphRAG+).

- **Day 2**: Hybrid DB setup (Neo4j/Qdrant/Redis, SQLite toggle); GraphRAG+ with agentic hybrid, local GPU embeddings (BGE-M3, int8 quantization, content-varied dims/chunks), Tavily/Exa integration for misses.

- **Day 3**: StateGraph for hierarchical shared state (short-term messages, long-term checkpointers, shared/private hybrid); hybrid task mgmt (deque + Redis Pub/Sub, centralized assign).

- **Day 4**: Multi-model routing (OpenRouter dynamic classify/intensity); 3-agent debate subgraph (pro/con/moderator, 2 rounds, voting/iteration).

- **Day 5**: Full autonomy flows (one-shot gen/feature PRs with ethical checks/tenacity); pytest unit/integration (mock DBs/simulate routing/debate); Docker Compose (services: Neo4j/Qdrant/Redis) with env toggles.

**Pseudocode for Phase 1 Main Workflow**:
```python
from langgraph.graph import StateGraph, add_messages
from typing import TypedDict, Annotated
from tools import graphrag_plus, route_model, debate_subgraph, task_queue

class State(TypedDict):
    messages: Annotated[list, add_messages]  # Shared scratchpad
    task_queue: deque  # Tasks
    private: dict  # Per-agent

workflow = StateGraph(State)
workflow.add_node('orchestrator', lambda state: task_queue.assign(state['input']))  # Task mgmt
workflow.add_node('research', lambda state: graphrag_plus(state['query']))  # GraphRAG+
workflow.add_node('debate', debate_subgraph)  # Debate
workflow.add_node('impl', lambda state: route_model(state['task'], 'coding'))  # Routing

# Edges: orchestrator --> research --> debate --> impl
graph = workflow.compile(checkpointer=MemorySaver())  # Persistence
```

### Phase 2: Extensions Implementation

**Overview**: Add multi-modal, advanced embeddings/debate, federated basics, scalability for toughest tasks/UI/web/privacy.

**Detailed Implementation**:

- **Day 1**: Multi-modal integration (openai SDK for CLIP embeddings in RAG/retrieval; process images in workflows for UI testing/validation, +20% acc).

- **Day 2**: Advanced embeddings (SPLADE sparse fusion with BGE-M3, dynamic by content for +15% precision in code/papers; toggle for perf).

- **Day 3**: Extended debate (toggle to 5 agents: add advocate/critic/refiner, +3 rounds for complex reasoning/planning/review, +10% gains).

- **Day 4**: Federated basics (Flower lib for local model aggregation in agent training; privacy-focused for multi-agent personalization without central data).

- **Day 5**: Enhanced scalability (Kubernetes opt-in for >100 agents; ZenRows web alt for deeper scrape if needed); full tests/benchmarks for Phase 2 features.

**Pseudocode for Phase 2 Multi-Modal Extension**:
```python
from openai import OpenAI  # SDK for vision

def multi_modal_rag(query, image_url=None):
    if image_url:
        vision_embed = OpenAI().embeddings.create(model='clip-vit-base-patch32', input=image_url)  # CLIP
        text_embed = embedder.encode(query)  # BGE-M3
        fused_embed = fuse_vision_text(vision_embed, text_embed)  # Custom concat/rank
        results = qdrant.search('collection', fused_embed)  # Hybrid retrieve
    else:
        results = graphrag_plus(query)  # Text-only fallback
    return results
```

**Pseudocode for Phase 2 Extended Debate**:
```python
def extended_debate(state, num_agents=3, rounds=2):  # Toggle
    if num_agents == 5:
        # Add advocate/critic/refiner nodes
        workflow.add_node('advocate', lambda s: llm("Advocate: " + s['task']))
        workflow.add_node('critic', lambda s: llm("Critic: " + s['task']))
        workflow.add_node('refiner', lambda s: llm("Refine: " + s['pro'] + s['con']))
        workflow.add_parallel(['pro', 'con', 'advocate', 'critic'])
        workflow.add_edge(['pro', 'con', 'advocate', 'critic'], 'refiner')
        workflow.add_edge('refiner', 'moderator')
        rounds = 3  # Extend
    # Existing 3-agent base
    return workflow.invoke(state, rounds=rounds)
```

## Research Reports and Performance Analysis

**Research Report**: Consolidated from all: GraphRAG+ (arXiv/Microsoft: 30-40% acc); routing (DataCamp/Vellum: 25% gains); debate (Du et al.: 30%); task/state (IBM/Medium: +30% eff). Phase 2: Multi-modal (arXiv: +20% UI acc); SPLADE (FastEmbed: +15% lexical); federated (Flower: privacy gains).

**Performance Analysis**: Phase 1: Latency <100ms, acc +30-40%; Phase 2: +20% multi-modal, +15% embeddings; costs <$50 add (cheap vision calls).

## Alternatives Considered

Across ADRs: E.g., CrewAI (simpler but less power, rejected for orchestration); Firecrawl (costly, rejected for web).
