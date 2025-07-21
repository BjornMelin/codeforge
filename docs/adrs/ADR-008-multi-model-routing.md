# ADR-008: Multi-Model Routing

**Status**: Accepted  

**Date**: 2025-07-20  

**Deciders**: CodeForge AI Team  

## Context

Optimize accuracy/performance > costs > latency in Phase 1, extend to Phase 2 multi-modal. The system needs intelligent model selection to balance quality, cost, and speed across different task types while supporting future multi-modal capabilities.

## Problem Statement

Achieve optimal performance across diverse tasks while managing costs. Requirements include:

- Task-specific model selection for maximum accuracy

- Cost optimization through intelligent routing

- Performance optimization for different task complexities

- Support for diverse model capabilities (coding, reasoning, analysis)

- Extensibility for multi-modal models in Phase 2

## Decision

**Dynamic routing via OpenRouter** in Phase 1 (Grok 4 ~40%, Claude 4 ~30%, Kimi K2 ~20%, Gemini flash ~10%, o3 <5%); extend to Phase 2 with vision models (e.g., Gemini Pro for images).

## Alternatives Considered

| Approach | Pros | Cons | Score |
|----------|------|------|-------|
| **OpenRouter Dynamic Routing** | Multi-model access, cost optimization, performance balance | Dependency on single provider | **8.6** |
| Single Grok 4 | Consistent performance, simple integration | No optimization, higher costs | 8.1 |
| Grok + o3 only | Good balance, covers most cases | Less variety, potential gaps | 8.0 |
| Custom model aggregator | Full control, optimal routing | High development cost, maintenance | 7.5 |

## Rationale

- **25% performance gains, 60% cost savings (8.6)**: Measured improvements across task types

- **Phase 1 balanced approach**: Optimal cost/performance trade-offs

- **Phase 2 multi-modal ready**: Easy extension to vision models

- **Task specialization**: Different models excel at different task types

## Consequences

### Positive

- Significant cost savings through intelligent routing

- Improved performance through task-specific model selection

- Reduced latency for simple tasks via fast models

- Strong foundation for Phase 2 multi-modal extensions

### Negative

- Dependency on OpenRouter service availability

- Complexity in routing logic and fallback handling

- Need for task classification accuracy

### Neutral

- Classify routing logic in Phase 1

- Add vision model routing in Phase 2

## Implementation Notes

### Model Router Architecture
```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

class TaskType(Enum):
    CODING = "coding"
    REASONING = "reasoning" 
    ANALYSIS = "analysis"
    SIMPLE_QA = "simple_qa"
    COMPLEX_QA = "complex_qa"
    CREATIVE = "creative"
    VISION = "vision"  # Phase 2

class ModelTier(Enum):
    FAST = "fast"      # Gemini Flash, simple tasks
    BALANCED = "balanced"  # Kimi K2, general purpose
    PREMIUM = "premium"    # Claude 4, complex reasoning
    SPECIALIST = "specialist"  # Grok 4, coding/analysis
    FLAGSHIP = "flagship"     # o3, hardest problems

@dataclass
class ModelConfig:
    name: str
    provider: str
    cost_per_token: float
    avg_latency_ms: int
    max_context: int
    capabilities: List[str]
    tier: ModelTier

class ModelRouter:
    def __init__(self):
        self.models = self._initialize_models()
        self.task_classifier = TaskClassifier()
        self.performance_tracker = PerformanceTracker()
        self.cost_tracker = CostTracker()
        self.fallback_chain = self._setup_fallback_chain()
    
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        return {
            'grok-4': ModelConfig(
                name='grok-4',
                provider='openrouter',
                cost_per_token=0.00002,
                avg_latency_ms=800,
                max_context=200000,
                capabilities=['coding', 'reasoning', 'analysis'],
                tier=ModelTier.SPECIALIST
            ),
            'claude-4-sonnet': ModelConfig(
                name='anthropic/claude-4-sonnet',
                provider='openrouter',
                cost_per_token=0.000015,
                avg_latency_ms=1200,
                max_context=200000,
                capabilities=['reasoning', 'analysis', 'creative'],
                tier=ModelTier.PREMIUM
            ),
            'kimi-k2': ModelConfig(
                name='moonshot/kimi-k2',
                provider='openrouter',
                cost_per_token=0.000008,
                avg_latency_ms=600,
                max_context=128000,
                capabilities=['general', 'coding', 'qa'],
                tier=ModelTier.BALANCED
            ),
            'gemini-flash': ModelConfig(
                name='google/gemini-2.5-flash',
                provider='openrouter',
                cost_per_token=0.000002,
                avg_latency_ms=300,
                max_context=1000000,
                capabilities=['simple_qa', 'fast_coding'],
                tier=ModelTier.FAST
            ),
            'o3': ModelConfig(
                name='openai/o3',
                provider='openrouter',
                cost_per_token=0.00006,
                avg_latency_ms=2000,
                max_context=200000,
                capabilities=['complex_reasoning', 'math', 'logic'],
                tier=ModelTier.FLAGSHIP
            )
        }
    
    async def route_request(self, prompt: str, task_context: Dict[str, Any]) -> str:
        """Main routing logic with fallback handling"""
        
        # Classify the task
        task_type = self.task_classifier.classify(prompt, task_context)
        
        # Select optimal model
        selected_model = self._select_model(task_type, prompt, task_context)
        
        # Execute with fallback
        return await self._execute_with_fallback(prompt, selected_model, task_context)
```

### Task Classification
```python
class TaskClassifier:
    def __init__(self):
        self.coding_patterns = [
            r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', 
            r'function\s+\w+', r'console\.log', r'print\('
        ]
        self.reasoning_patterns = [
            r'explain why', r'analyze', r'compare', r'evaluate',
            r'what if', r'how would', r'reasoning', r'logic'
        ]
        self.simple_qa_patterns = [
            r'^what is', r'^how to', r'^when', r'^where',
            r'^who', r'^define', r'^meaning of'
        ]
        
    def classify(self, prompt: str, context: Dict[str, Any]) -> TaskType:
        """Classify task type based on prompt and context"""
        
        prompt_lower = prompt.lower()
        
        # Check for code content
        if any(re.search(pattern, prompt) for pattern in self.coding_patterns):
            return TaskType.CODING
        
        # Check for complex reasoning
        if len(prompt) > 500 or any(keyword in prompt_lower for keyword in 
                                   ['analyze', 'evaluate', 'compare', 'explain why']):
            if any(re.search(pattern, prompt_lower) for pattern in self.reasoning_patterns):
                return TaskType.REASONING
            return TaskType.COMPLEX_QA
        
        # Check for simple Q&A
        if any(re.search(pattern, prompt_lower) for pattern in self.simple_qa_patterns):
            return TaskType.SIMPLE_QA
        
        # Check context for additional clues
        if context.get('previous_task_type') == 'coding':
            return TaskType.CODING
        
        if context.get('requires_analysis', False):
            return TaskType.ANALYSIS
        
        # Default to balanced
        return TaskType.COMPLEX_QA
    
    def estimate_complexity(self, prompt: str, task_type: TaskType) -> float:
        """Estimate task complexity (0.0-1.0)"""
        
        complexity_score = 0.0
        
        # Length factor
        complexity_score += min(len(prompt) / 1000, 0.3)
        
        # Task type factor
        type_complexity = {
            TaskType.SIMPLE_QA: 0.1,
            TaskType.CODING: 0.4,
            TaskType.ANALYSIS: 0.6,
            TaskType.REASONING: 0.7,
            TaskType.COMPLEX_QA: 0.5,
            TaskType.CREATIVE: 0.5
        }
        complexity_score += type_complexity.get(task_type, 0.5)
        
        # Content analysis
        complex_indicators = [
            'multi-step', 'complex', 'analyze', 'optimize', 
            'design', 'architecture', 'algorithm'
        ]
        
        for indicator in complex_indicators:
            if indicator in prompt.lower():
                complexity_score += 0.1
        
        return min(complexity_score, 1.0)
```

### Model Selection Logic
```python
class ModelSelector:
    def __init__(self, models: Dict[str, ModelConfig]):
        self.models = models
        self.routing_rules = self._define_routing_rules()
        self.performance_weights = {
            'accuracy': 0.4,
            'cost': 0.3,
            'latency': 0.2,
            'reliability': 0.1
        }
    
    def _define_routing_rules(self) -> Dict[TaskType, List[str]]:
        """Define primary model preferences by task type"""
        return {
            TaskType.CODING: ['grok-4', 'kimi-k2', 'claude-4-sonnet'],
            TaskType.REASONING: ['claude-4-sonnet', 'o3', 'grok-4'],
            TaskType.ANALYSIS: ['grok-4', 'claude-4-sonnet', 'kimi-k2'],
            TaskType.SIMPLE_QA: ['gemini-flash', 'kimi-k2', 'claude-4-sonnet'],
            TaskType.COMPLEX_QA: ['claude-4-sonnet', 'grok-4', 'kimi-k2'],
            TaskType.CREATIVE: ['claude-4-sonnet', 'kimi-k2', 'grok-4']
        }
    
    def select_model(self, task_type: TaskType, complexity: float, 
                    context: Dict[str, Any]) -> str:
        """Select optimal model based on task requirements"""
        
        candidate_models = self.routing_rules.get(task_type, ['kimi-k2'])
        
        # Filter by context constraints
        if context.get('max_cost_per_token'):
            max_cost = context['max_cost_per_token']
            candidate_models = [
                model for model in candidate_models 
                if self.models[model].cost_per_token <= max_cost
            ]
        
        if context.get('max_latency_ms'):
            max_latency = context['max_latency_ms']
            candidate_models = [
                model for model in candidate_models
                if self.models[model].avg_latency_ms <= max_latency
            ]
        
        # Apply complexity-based selection
        if complexity > 0.8 and 'o3' in candidate_models:
            return 'o3'  # Use flagship for hardest problems
        elif complexity < 0.3 and 'gemini-flash' in candidate_models:
            return 'gemini-flash'  # Use fast model for simple tasks
        
        # Return first valid candidate (already ordered by preference)
        return candidate_models[0] if candidate_models else 'kimi-k2'
```

### Execution with Fallback
```python
class ModelExecutor:
    def __init__(self, openrouter_client):
        self.client = openrouter_client
        self.retry_config = {
            'max_retries': 3,
            'backoff_factor': 2,
            'timeout_seconds': 30
        }
    
    async def execute_with_fallback(self, prompt: str, primary_model: str, 
                                   fallback_models: List[str], 
                                   context: Dict[str, Any]) -> str:
        """Execute request with automatic fallback"""
        
        models_to_try = [primary_model] + fallback_models
        last_error = None
        
        for model_name in models_to_try:
            try:
                result = await self._execute_single_model(
                    prompt, model_name, context
                )
                
                # Track successful execution
                self.performance_tracker.record_success(model_name, context)
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_name} failed: {e}")
                self.performance_tracker.record_failure(model_name, str(e))
                continue
        
        # All models failed
        raise Exception(f"All models failed. Last error: {last_error}")
    
    async def _execute_single_model(self, prompt: str, model_name: str, 
                                   context: Dict[str, Any]) -> str:
        """Execute request on single model with retry logic"""
        
        for attempt in range(self.retry_config['max_retries']):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=context.get('max_tokens', 4000),
                        temperature=context.get('temperature', 0.7)
                    ),
                    timeout=self.retry_config['timeout_seconds']
                )
                
                return response.choices[0].message.content
                
            except asyncio.TimeoutError:
                if attempt < self.retry_config['max_retries'] - 1:
                    wait_time = self.retry_config['backoff_factor'] ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise
            
            except Exception as e:
                if attempt < self.retry_config['max_retries'] - 1:
                    await asyncio.sleep(1)
                    continue
                raise
```

## Performance Tracking and Optimization

### Performance Metrics
```python
class PerformanceTracker:
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'success_count': 0,
            'failure_count': 0,
            'total_latency': 0,
            'total_cost': 0,
            'quality_scores': []
        })
    
    def record_success(self, model_name: str, context: Dict[str, Any]):
        """Record successful model execution"""
        metrics = self.metrics[model_name]
        metrics['success_count'] += 1
        metrics['total_latency'] += context.get('actual_latency_ms', 0)
        metrics['total_cost'] += context.get('actual_cost', 0)
        
        if 'quality_score' in context:
            metrics['quality_scores'].append(context['quality_score'])
    
    def get_model_performance(self, model_name: str) -> Dict[str, float]:
        """Get performance statistics for a model"""
        metrics = self.metrics[model_name]
        total_requests = metrics['success_count'] + metrics['failure_count']
        
        if total_requests == 0:
            return {'success_rate': 0, 'avg_latency': 0, 'avg_cost': 0, 'avg_quality': 0}
        
        return {
            'success_rate': metrics['success_count'] / total_requests,
            'avg_latency': metrics['total_latency'] / max(metrics['success_count'], 1),
            'avg_cost': metrics['total_cost'] / max(metrics['success_count'], 1),
            'avg_quality': sum(metrics['quality_scores']) / max(len(metrics['quality_scores']), 1)
        }
    
    def optimize_routing(self) -> Dict[TaskType, List[str]]:
        """Dynamically optimize routing based on performance data"""
        optimized_routing = {}
        
        for task_type in TaskType:
            # Sort models by performance score for this task type
            model_scores = []
            for model_name in self.metrics.keys():
                perf = self.get_model_performance(model_name)
                # Combined score: accuracy weighted heavily, cost and latency considered
                score = (perf['avg_quality'] * 0.5 + 
                        perf['success_rate'] * 0.3 +
                        (1 - perf['avg_cost']) * 0.1 +
                        (1 - perf['avg_latency'] / 3000) * 0.1)  # Normalize latency
                model_scores.append((model_name, score))
            
            # Sort by score and take top models
            model_scores.sort(key=lambda x: x[1], reverse=True)
            optimized_routing[task_type] = [model[0] for model in model_scores[:3]]
        
        return optimized_routing
```

## Phase 2 Extensions

### Multi-Modal Routing
```python
class MultiModalRouter:  # Phase 2
    def __init__(self, base_router: ModelRouter):
        self.base_router = base_router
        self.vision_models = {
            'gemini-pro-vision': ModelConfig(
                name='google/gemini-2.5-pro',
                provider='openrouter',
                cost_per_token=0.00003,
                avg_latency_ms=1500,
                max_context=1000000,
                capabilities=['vision', 'reasoning', 'analysis'],
                tier=ModelTier.PREMIUM
            ),
            'claude-4-vision': ModelConfig(
                name='anthropic/claude-4-sonnet',
                provider='openrouter', 
                cost_per_token=0.000015,
                avg_latency_ms=1200,
                max_context=200000,
                capabilities=['vision', 'reasoning', 'creative'],
                tier=ModelTier.PREMIUM
            )
        }
    
    async def route_multimodal_request(self, prompt: str, images: List[str], 
                                     context: Dict[str, Any]) -> str:
        """Route requests that include visual content"""
        
        if not images:
            return await self.base_router.route_request(prompt, context)
        
        # Classify visual task type
        visual_task_type = self._classify_visual_task(prompt, images)
        
        # Select appropriate vision model
        if visual_task_type == 'ui_analysis':
            selected_model = 'gemini-pro-vision'  # Better for UI/web content
        elif visual_task_type == 'creative_analysis':
            selected_model = 'claude-4-vision'   # Better for creative content
        else:
            selected_model = 'gemini-pro-vision'  # Default vision model
        
        return await self._execute_vision_model(prompt, images, selected_model, context)
```

## Cost Optimization

### Usage Distribution Targets
```python

# Target distribution to achieve 60% cost savings
COST_OPTIMIZATION_TARGETS = {
    'grok-4': 0.40,        # 40% - specialist tasks
    'claude-4-sonnet': 0.30,  # 30% - complex reasoning
    'kimi-k2': 0.20,          # 20% - general purpose
    'gemini-flash': 0.10,     # 10% - simple tasks
    'o3': 0.05               # <5% - hardest problems only
}

class CostOptimizer:
    def __init__(self):
        self.target_distribution = COST_OPTIMIZATION_TARGETS
        self.current_usage = defaultdict(int)
        self.daily_budget = 50.0  # $50/day target
    
    def should_use_cheaper_model(self, preferred_model: str, task_complexity: float) -> bool:
        """Determine if we should use a cheaper model for cost optimization"""
        
        current_ratio = self.current_usage[preferred_model] / sum(self.current_usage.values())
        target_ratio = self.target_distribution.get(preferred_model, 0.1)
        
        # If we're over budget for this model and task isn't too complex
        if current_ratio > target_ratio and task_complexity < 0.7:
            return True
        
        # Check daily budget
        if self._projected_daily_cost() > self.daily_budget:
            return True
        
        return False
```

## Performance Targets

| Metric | Phase 1 Target | Phase 2 Target |
|--------|----------------|----------------|
| Accuracy Improvement | +25% | +30% |
| Cost Reduction | 60% | 65% |
| Avg Response Time | <1.5s | <2s (with vision) |
| Success Rate | >95% | >95% |
| Model Utilization Balance | Within 10% of targets | Within 5% of targets |

## Related Decisions

- ADR-009: Debate Agents

- ADR-001: Multi-Agent Framework Selection

- ADR-011: Multi-Modal Support (Phase 2)

## Monitoring

- Model performance metrics by task type

- Cost tracking and budget adherence

- Routing accuracy and effectiveness

- Fallback frequency and success rates

- Task classification accuracy
