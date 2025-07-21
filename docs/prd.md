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

1. **Autonomous Orchestration**: LangGraph v0.5.3+ graphs for workflows; hierarchical agents with 3-agent debate (proponent, opponent, moderator).
2. **SOTA Retrieval**: GraphRAG+ with Neo4j v5.28.1+ and Qdrant v1.15.0+, BGE-M3 embeddings with int8 quantization, content-aware dimensions (384D code, 768D docs, 256D functions), Tavily v0.7.10+ primary with Exa fallback.
3. **Multi-Model Routing**: OpenRouter dynamic (xai/grok-4 ~40%, anthropic/claude-4-sonnet ~30%, kimi/k2 ~20%, google/gemini-2.5-flash ~10%, o3 <5%).
4. **Task Management**: LangGraph v0.5.3+ StateGraph with Redis v6.0.0+ Pub/Sub, hybrid in-memory deque + persistent coordination.
5. **Shared State**: LangGraph v0.5.3+ StateGraph hierarchical memory (short-term in-memory <1ms, long-term SQLite/DB).
6. **Enhanced Tools**: Direct SDK integration for core tools (qdrant-client v1.15.0+, neo4j v5.28.1+, redis v6.0.0+), MCP for custom tools.
7. **Autonomy Flows**: One-shot gen/feature PRs with 3-agent debate/review, max 2 rounds for efficiency.

### Phase 2 Features (Optimizations/Extensions)

1. **Multi-Modal Support**: Vision integration with OpenAI SDK v1.97.0+ and GPT-4V, CLIP embeddings for visual search, support for UI/web development from screenshots.
2. **Advanced Embeddings**: SPLADE sparse embeddings via sentence-transformers v5.0.0+, hybrid with BGE-M3 dense for +15% precision, weighted fusion optimization.
3. **Extended Debate**: 5-agent configuration (proponent, opponent, advocate, critic, moderator), up to 3 rounds for complex decisions, specialized role assignment.
4. **Federated Basics**: Flower v1.12.1+ framework with differential privacy, secure aggregation for collaborative learning, privacy-preserving model improvements.
5. **Enhanced Scalability**: Redis Cluster v6.0.0+ for distributed state, Kubernetes orchestration, auto-scaling based on demand, multi-zone deployment.

## Functional Requirements

### Phase 1 Functional Requirements

| ID | Feature | Description | Priority | Acceptance Criteria |
|----|---------|-------------|----------|---------------------|
| FR-01 | Orchestration | LangGraph v0.5.3+ StateGraph with conditional routing and subgraphs for 3-agent debate. | High | Executes workflows with <100ms overhead; debate improves accuracy 30-40%. |
| FR-02 | Retrieval | GraphRAG+ hybrid with Neo4j graph traversal + Qdrant vector search, BGE-M3 embeddings with int8 quantization. | High | 30% accuracy improvement over baseline RAG; <500ms query latency. |
| FR-03 | Model Routing | Dynamic model selection via OpenRouter based on task complexity and category. | High | 25% performance improvement; <$100/month cost target. |
| FR-04 | Task Mgmt | Hybrid in-memory deque + Redis Pub/Sub with dependency resolution. | High | <50ms task assignment; support 200+ concurrent tasks. |
| FR-05 | Shared State | Hierarchical memory with LangGraph StateGraph, anti-hallucination mechanisms. | High | 100% state consistency; <1ms short-term access. |
| FR-06 | Tools | Direct SDK integration with <10ms Redis, <50ms Qdrant, <100ms Neo4j latency. | High | >99% tool availability; automatic retry on failure. |
| FR-07 | Autonomy | Complete workflow from PRD to PR with debate validation. | High | 95% task completion rate; ethical checks included. |

### Phase 2 Functional Requirements

| ID | Feature | Description | Priority | Acceptance Criteria |
|----|---------|-------------|----------|---------------------|
| FR-08 | Multi-Modal | GPT-4V integration for UI analysis, CLIP embeddings for visual search. | Medium | >85% UI element recognition; <10s processing time. |
| FR-09 | Advanced Embeddings | SPLADE sparse + BGE-M3 dense hybrid with adaptive weighting. | Medium | 15% precision improvement; <800ms hybrid search. |
| FR-10 | Extended Debate | 5-agent system with specialized roles and parallel processing. | Medium | 50% improvement for complex decisions; <180s total time. |
| FR-11 | Federated Basics | Flower framework with ε-differential privacy (ε ≤ 1.0). | Low | Zero raw data transmission; 10-15% model improvement. |
| FR-12 | Scalability | Redis Cluster + Kubernetes with auto-scaling and health monitoring. | Low | Support 100+ instances; <2min scaling response. |

## Non-Functional Requirements

### Phase 1 Non-Functional Requirements

| ID | Category | Requirement | Metric | Validation |
|----|----------|-------------|--------|------------|
| NFR-01 | Performance | Response latency and accuracy | <100ms orchestration overhead, 30% accuracy gain | Load testing and benchmarks |
| NFR-02 | Cost | Monthly operational cost | <$200 total, ~$3-5 daily budget | OpenRouter usage tracking |
| NFR-03 | Maintainability | Library-first approach | Direct SDK usage, minimal custom code | Code review metrics |
| NFR-04 | Scalability | Concurrent agent support | 10+ agents Phase 1, 50+ Phase 2 ready | Docker Compose testing |
| NFR-05 | Reliability | System uptime | >99.5% availability, <30s failover | Health monitoring |

### Phase 2 Non-Functional Requirements

| ID | Category | Requirement | Metric | Validation |
|----|----------|-------------|--------|------------|
| NFR-06 | Performance | Multi-modal accuracy | >85% UI recognition, >80% layout analysis | Vision benchmarks |
| NFR-07 | Cost | Additional features | <$50/month incremental cost | Usage monitoring |
| NFR-08 | Maintainability | Feature toggles | Zero-impact when disabled | Integration tests |
| NFR-09 | Scalability | Extended capacity | 100+ concurrent users per instance | K8s load testing |
| NFR-10 | Privacy | Federated learning | ε ≤ 1.0 differential privacy | Privacy audits |

## Scope & Prioritization

- **Phase 1 (MVP)**: Core autonomy with orchestration, GraphRAG+, routing, 3-agent debate, task/state management - 3-5 days implementation.

- **Phase 2**: Extensions including multi-modal, advanced embeddings, 5-agent debate, federated basics - 1 week post-MVP; prioritize multi-modal first for UI/web developers.

- **Out of Scope**: Full enterprise federated learning, custom LLM fine-tuning, production monitoring dashboards.

## Dependencies & Integrations

### Phase 1 Dependencies

- **Frameworks**: LangGraph v0.5.3+ (orchestration and state management)
- **Models**: OpenRouter API (xai/grok-4, anthropic/claude-4-sonnet, kimi/k2, google/gemini-2.5-flash, o3)
- **Databases**: Neo4j v5.28.1+ (graph), Qdrant v1.15.0+ (vector), Redis v6.0.0+ (cache/pubsub)
- **Embeddings**: sentence-transformers v5.0.0+ (BGE-M3 with int8 quantization)
- **Search**: Tavily v0.7.10+ (primary), Exa (fallback)
- **Core Libs**: Python 3.12+, httpx v0.28.0+, tenacity v9.1.2+, pydantic v2.11.7+, uv (package manager)

### Phase 2 Dependencies

- **Vision**: OpenAI SDK v1.97.0+ (GPT-4V integration)
- **Embeddings**: SPLADE via sentence-transformers v5.0.0+
- **Federated**: Flower v1.12.1+ (privacy-preserving learning)
- **Scaling**: Redis Cluster v6.0.0+, Kubernetes
- **Optional**: ZenRows (advanced web scraping), PyTorch v2.7.1+ (GPU acceleration)

## Risks & Mitigations

### Phase 1 Risks

- **Model routing latency**: Implement caching and pre-classification of common patterns
- **State synchronization overhead**: Use hierarchical memory with careful message capping
- **Cost overruns**: Route 70% to cheaper models (Gemini Flash, Kimi K2), monitor usage

### Phase 2 Risks

- **Multi-modal processing delays**: Implement async processing and result caching
- **Federated privacy concerns**: Start with basic aggregation only, extensive testing
- **Scaling complexity**: Begin with Docker Compose, gradual migration to K8s

## Success Metrics

### Phase 1 Success Metrics

- Task completion rate: >95%
- Orchestration latency: <100ms
- Retrieval accuracy improvement: 30-40%
- Monthly cost: <$200
- Hallucination rate: <10%
- Debate effectiveness: 30% accuracy improvement

### Phase 2 Success Metrics

- Multi-modal UI recognition: >85%
- Hybrid embedding precision: +15%
- Extended debate quality: +50% for complex decisions
- Federated model improvement: 10-15%
- Horizontal scalability: 100+ instances

## Timeline

### Phase 1: MVP (3-5 Days)

- **Day 1**: Install deps with uv, implement tools.py with SDK integrations
- **Day 2**: Set up Neo4j/Qdrant/Redis, implement GraphRAG+ with BGE-M3
- **Day 3**: Implement LangGraph StateGraph and Redis task management
- **Day 4**: Add OpenRouter model routing and 3-agent debate system
- **Day 5**: Complete autonomy workflows, testing, Docker Compose deployment

### Phase 2: Extensions (1 Week Post-MVP)

- **Day 1**: Integrate OpenAI SDK for vision, add CLIP embeddings
- **Day 2**: Implement SPLADE sparse embeddings with fusion
- **Day 3**: Extend to 5-agent debate with specialized roles
- **Day 4**: Add Flower framework for federated basics
- **Day 5**: Implement Kubernetes configs and scalability testing

## Implementation Plans

### Phase 1: MVP Implementation

**Overview**: Build core autonomy with LangGraph v0.5.3+, hybrid databases, intelligent routing, 3-agent debate, and shared state management.

**Detailed Implementation**:

- **Day 1**: Set up project with uv package manager. Install dependencies: LangGraph v0.5.3+, sentence-transformers v5.0.0+, OpenRouter SDK, database clients (neo4j v5.28.1+, qdrant-client v1.15.0+, redis v6.0.0+). Implement tools.py with direct SDK integrations for low-latency operations.

- **Day 2**: Deploy Neo4j/Qdrant/Redis via Docker Compose. Implement GraphRAG+ with BGE-M3 embeddings (int8 quantization), content-aware dimensions (384D code, 768D docs, 256D functions). Add Tavily/Exa web search integration for RAG misses.

- **Day 3**: Implement LangGraph StateGraph for hierarchical memory (short-term in-memory, long-term persistent). Set up hybrid task management with in-memory deque + Redis Pub/Sub for coordination.

- **Day 4**: Add OpenRouter integration with dynamic model routing based on task complexity. Implement 3-agent debate subgraph (proponent, opponent, moderator) with 2-round maximum and consensus voting.

- **Day 5**: Complete end-to-end autonomy workflows from PRD to PR. Add comprehensive pytest suite with mocked services. Finalize Docker Compose configuration with environment toggles.

**Pseudocode for Phase 1 Main Workflow**:

```python
from langgraph.graph import StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from tools import graphrag_plus, route_model, debate_subgraph
from collections import deque

class State(TypedDict):
    messages: Annotated[list, add_messages]  # Shared context
    task_queue: deque[str]  # Task queue
    private: dict  # Per-agent state
    long_term: dict  # Persistent memory

workflow = StateGraph(State)
workflow.add_node('assign_task', lambda s: task_queue.assign(s['input']))
workflow.add_node('research', lambda s: graphrag_plus(s['task']))
workflow.add_node('debate', debate_subgraph)  # 3-agent subgraph
workflow.add_node('implement', lambda s: route_model(s['task'], 'coding'))

# Connect workflow
workflow.add_edge('assign_task', 'research')
workflow.add_edge('research', 'debate')
workflow.add_edge('debate', 'implement')

graph = workflow.compile(checkpointer=MemorySaver())
```

### Phase 2: Extensions Implementation

**Overview**: Add multi-modal vision, advanced embeddings, extended debate, federated learning, and horizontal scalability.

**Detailed Implementation**:

- **Day 1**: Integrate OpenAI SDK v1.97.0+ for GPT-4V vision analysis. Add CLIP embeddings to GraphRAG+ for visual similarity search. Process UI screenshots and generate appropriate code.

- **Day 2**: Implement SPLADE sparse embeddings alongside BGE-M3 dense embeddings. Add weighted fusion with adaptive weighting based on query type. Achieve 15% precision improvement.

- **Day 3**: Extend debate system to 5 agents (add advocate and critic roles). Implement parallel initial argument phase for efficiency. Support up to 3 rounds for complex decisions.

- **Day 4**: Integrate Flower v1.12.1+ for federated learning basics. Implement differential privacy with ε ≤ 1.0. Enable privacy-preserving model improvements across instances.

- **Day 5**: Add Redis Cluster support for distributed state. Create Kubernetes deployment configs with auto-scaling. Implement comprehensive monitoring and health checks.

**Pseudocode for Phase 2 Multi-Modal Extension**:

```python
from openai import OpenAI
import numpy as np

class MultiModalRAG:
    def __init__(self):
        self.client = OpenAI()
        self.embedder = SentenceTransformer('BAAI/bge-m3')
    
    async def process_ui_image(self, image_path: str, query: str):
        # Analyze UI with GPT-4V
        response = await self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": image_path}}
                ]
            }]
        )
        
        # Generate CLIP embeddings for visual search
        visual_embedding = self.generate_clip_embedding(image_path)
        text_embedding = self.embedder.encode(query)
        
        # Hybrid retrieval
        return self.hybrid_search(visual_embedding, text_embedding)
```

**Pseudocode for Phase 2 Extended Debate**:

```python
def create_extended_debate(num_agents: int = 5):
    debate = StateGraph(State)
    
    # Core agents
    debate.add_node('proponent', pro_agent)
    debate.add_node('opponent', con_agent)
    debate.add_node('moderator', moderator_agent)
    
    if num_agents == 5:
        # Extended agents
        debate.add_node('advocate', advocate_agent)  # User perspective
        debate.add_node('critic', critic_agent)      # Technical analysis
        
        # Parallel initial arguments
        debate.add_parallel(['proponent', 'opponent', 'advocate', 'critic'])
        debate.add_edge(['proponent', 'opponent', 'advocate', 'critic'], 'moderator')
    
    return debate.compile()
```

## Research Reports and Performance Analysis

**Research Report Summary**:

- GraphRAG (Microsoft Research): 30-40% accuracy improvement through graph+vector fusion
- Multi-agent debate (Du et al., 2023): 30% reduction in hallucinations
- Sparse+dense embeddings (SPLADE): 15% precision gain for technical queries
- Federated learning (Flower): Privacy-preserving improvements without data sharing
- Model routing (industry studies): 25% performance gain through specialization

**Performance Analysis**:

- Phase 1: <100ms latency with 30-40% accuracy gains at <$200/month
- Phase 2: Additional 20% multi-modal accuracy, 15% embedding precision for <$50/month incremental cost

## Alternatives Considered

- **Orchestration**: Rejected CrewAI (less flexible graph control), AutoGPT (poor state management)
- **Vector DB**: Rejected Pinecone (cost), Weaviate (complexity vs Qdrant)
- **Web Search**: Rejected Firecrawl (expensive), SerpAPI (limited content extraction)
- **Embeddings**: Rejected OpenAI embeddings (cost), Cohere (less accurate than BGE-M3)
- **Federated**: Rejected full blockchain (complexity), central server (privacy concerns)
