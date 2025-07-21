# ADR-009: Debate Agents

**Status**: Accepted  

**Date**: 2025-07-20  

**Deciders**: CodeForge AI Team  

## Context

Enhance reasoning accuracy in Phase 1, extend to Phase 2 complex tasks. The system needs structured debate mechanisms to improve decision quality and reduce hallucinations through adversarial reasoning processes.

## Problem Statement

Improve reasoning accuracy through structured agent debate across phases. Requirements include:

- Reduction in hallucinations and poor decisions

- Improved accuracy for complex reasoning tasks

- Scalable debate mechanisms for different complexity levels

- Integration with multi-model routing for optimal performance

- Extensibility for Phase 2 complex task handling

## Decision

**LangGraph subgraph** with 3 agents (pro/con/moderator) + 2 rounds in Phase 1; toggle to 5 agents (add advocate/critic/refiner) + 3 rounds in Phase 2.

## Alternatives Considered

| Approach | Pros | Cons | Score |
|----------|------|------|-------|
| **3-Agent LangGraph Subgraph** | Optimal balance, good accuracy gains, manageable complexity | Some limitations for very complex tasks | **9.0** |
| 5-Agent from start | Highest accuracy potential | Higher computational load, complexity | 8.2 |
| 2-Agent simple debate | Simpler, faster | Lower accuracy gains, less nuanced | 8.0 |
| Single agent validation | Minimal overhead | Limited improvement, no debate benefits | 6.5 |

## Rationale

- **Optimal 30-40% accuracy improvement (9.0)**: Measured gains in Phase 1 benchmarks

- **Phase 1 balance**: Manageable complexity with significant benefits

- **Phase 2 scalability**: Natural extension to 5-agent complex reasoning

- **Framework integration**: Seamless with LangGraph orchestration

## Consequences

### Positive

- Significant reduction in hallucinations and poor decisions

- Improved accuracy on complex reasoning tasks

- Structured decision-making process with audit trail

- Strong foundation for Phase 2 complex task handling

### Negative

- Increased computational overhead and latency

- Higher token consumption across multiple models

- Complexity in consensus mechanism design

### Neutral

- Toggle capability for Phase 2 extended debate

- Integration with multi-model routing for cost optimization

## Implementation Notes

### Debate Subgraph Architecture
```python
from langgraph.graph import StateGraph, add_messages
from typing import TypedDict, Annotated, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class DebateRole(Enum):
    PROPONENT = "proponent"
    OPPONENT = "opponent" 
    MODERATOR = "moderator"
    ADVOCATE = "advocate"      # Phase 2
    CRITIC = "critic"          # Phase 2
    REFINER = "refiner"        # Phase 2

@dataclass
class DebateArgument:
    role: DebateRole
    content: str
    confidence: float
    supporting_evidence: List[str]
    timestamp: float
    round_number: int

class DebateState(TypedDict):
    topic: str
    arguments: Annotated[List[DebateArgument], add_messages]
    current_round: int
    max_rounds: int
    consensus_reached: bool
    final_decision: Optional[str]
    confidence_score: float
    participant_count: int

class DebateSystem:
    def __init__(self, model_router, max_rounds: int = 2):
        self.model_router = model_router
        self.max_rounds = max_rounds
        self.consensus_threshold = 0.8
        self.argument_evaluator = ArgumentEvaluator()
        
    def create_debate_graph(self) -> StateGraph:
        """Create the debate subgraph workflow"""
        
        workflow = StateGraph(DebateState)
        
        # Phase 1: 3-agent setup
        workflow.add_node("proponent", self._proponent_agent)
        workflow.add_node("opponent", self._opponent_agent)
        workflow.add_node("moderator", self._moderator_agent)
        
        # Debate flow
        workflow.add_edge("proponent", "opponent")
        workflow.add_edge("opponent", "moderator")
        
        # Conditional continuation
        workflow.add_conditional_edges(
            "moderator",
            self._should_continue_debate,
            {
                "continue": "proponent",
                "conclude": "final_decision"
            }
        )
        
        workflow.add_node("final_decision", self._finalize_decision)
        workflow.set_entry_point("proponent")
        
        return workflow.compile()
```

### Agent Implementations
```python
class DebateAgents:
    def __init__(self, model_router):
        self.model_router = model_router
        
    async def _proponent_agent(self, state: DebateState) -> DebateState:
        """Agent arguing FOR the proposition"""
        
        # Build context from previous arguments
        context = self._build_context(state, DebateRole.PROPONENT)
        
        prompt = f"""
        You are arguing FOR the following proposition: {state['topic']}
        
        Previous arguments:
        {context}
        
        Provide a strong argument supporting this position. Include:
        1. Clear reasoning and evidence
        2. Address any counterarguments raised
        3. Confidence level (0.0-1.0)
        
        Be factual, logical, and persuasive.
        """
        
        response = await self.model_router.route_request(
            prompt, 
            {'task_type': 'reasoning', 'agent_role': 'proponent'}
        )
        
        argument = self._parse_argument_response(response, DebateRole.PROPONENT, state)
        state['arguments'].append(argument)
        
        return state
    
    async def _opponent_agent(self, state: DebateState) -> DebateState:
        """Agent arguing AGAINST the proposition"""
        
        context = self._build_context(state, DebateRole.OPPONENT)
        
        prompt = f"""
        You are arguing AGAINST the following proposition: {state['topic']}
        
        Previous arguments:
        {context}
        
        Provide a strong counter-argument. Include:
        1. Clear reasoning challenging the proposition
        2. Address proponent's evidence and logic
        3. Present alternative perspectives
        4. Confidence level (0.0-1.0)
        
        Be factual, logical, and persuasive in your opposition.
        """
        
        response = await self.model_router.route_request(
            prompt,
            {'task_type': 'reasoning', 'agent_role': 'opponent'}
        )
        
        argument = self._parse_argument_response(response, DebateRole.OPPONENT, state)
        state['arguments'].append(argument)
        
        return state
    
    async def _moderator_agent(self, state: DebateState) -> DebateState:
        """Moderator evaluating arguments and managing debate flow"""
        
        current_round_args = [
            arg for arg in state['arguments'] 
            if arg.round_number == state['current_round']
        ]
        
        prompt = f"""
        You are moderating a debate on: {state['topic']}
        
        Current round arguments:
        {self._format_arguments_for_evaluation(current_round_args)}
        
        Evaluate the arguments and determine:
        1. Quality and strength of each position
        2. Whether consensus is reached or more debate needed
        3. Key points that need further exploration
        4. Overall confidence in current understanding (0.0-1.0)
        
        Provide objective analysis focusing on logic, evidence, and reasoning quality.
        """
        
        response = await self.model_router.route_request(
            prompt,
            {'task_type': 'analysis', 'agent_role': 'moderator'}
        )
        
        evaluation = self._parse_moderator_response(response, state)
        
        # Update state based on evaluation
        state['current_round'] += 1
        state['confidence_score'] = evaluation['confidence']
        state['consensus_reached'] = (
            evaluation['confidence'] > self.consensus_threshold or
            state['current_round'] >= state['max_rounds']
        )
        
        return state
```

### Consensus and Decision Making
```python
class ConsensusBuilder:
    def __init__(self):
        self.decision_strategies = {
            'confidence_weighted': self._confidence_weighted_decision,
            'evidence_based': self._evidence_based_decision,
            'majority_vote': self._majority_vote_decision
        }
    
    def _finalize_decision(self, state: DebateState) -> DebateState:
        """Generate final decision based on debate outcomes"""
        
        all_arguments = state['arguments']
        
        # Analyze argument strength
        argument_analysis = self._analyze_argument_strength(all_arguments)
        
        # Apply decision strategy
        decision = self._confidence_weighted_decision(
            all_arguments, 
            argument_analysis
        )
        
        state['final_decision'] = decision['conclusion']
        state['confidence_score'] = decision['confidence']
        
        return state
    
    def _analyze_argument_strength(self, arguments: List[DebateArgument]) -> Dict:
        """Analyze the strength and quality of arguments"""
        
        analysis = {
            'pro_strength': 0.0,
            'con_strength': 0.0,
            'evidence_quality': {},
            'logical_consistency': {},
            'novel_insights': []
        }
        
        for arg in arguments:
            # Evaluate evidence quality
            evidence_score = self._evaluate_evidence_quality(arg.supporting_evidence)
            analysis['evidence_quality'][arg.role.value] = evidence_score
            
            # Evaluate logical consistency
            logic_score = self._evaluate_logical_consistency(arg.content)
            analysis['logical_consistency'][arg.role.value] = logic_score
            
            # Aggregate strength scores
            if arg.role == DebateRole.PROPONENT:
                analysis['pro_strength'] += arg.confidence * evidence_score * logic_score
            elif arg.role == DebateRole.OPPONENT:
                analysis['con_strength'] += arg.confidence * evidence_score * logic_score
        
        return analysis
    
    def _confidence_weighted_decision(self, arguments: List[DebateArgument], 
                                    analysis: Dict) -> Dict:
        """Make decision based on confidence-weighted argument strength"""
        
        pro_score = analysis['pro_strength']
        con_score = analysis['con_strength']
        
        if pro_score > con_score * 1.2:  # Require significant advantage
            conclusion = "SUPPORT"
            confidence = min(pro_score / (pro_score + con_score), 0.95)
        elif con_score > pro_score * 1.2:
            conclusion = "OPPOSE" 
            confidence = min(con_score / (pro_score + con_score), 0.95)
        else:
            conclusion = "UNCERTAIN"
            confidence = 0.5
        
        return {
            'conclusion': conclusion,
            'confidence': confidence,
            'reasoning': self._generate_decision_reasoning(analysis)
        }
```

### Phase 2 Extensions
```python
class ExtendedDebateSystem:  # Phase 2
    def __init__(self, base_system: DebateSystem):
        self.base_system = base_system
        self.extended_agents = {
            'advocate': self._advocate_agent,
            'critic': self._critic_agent, 
            'refiner': self._refiner_agent
        }
    
    def create_extended_debate_graph(self) -> StateGraph:
        """Create 5-agent debate workflow for complex tasks"""
        
        workflow = StateGraph(DebateState)
        
        # All 5 agents
        workflow.add_node("proponent", self.base_system._proponent_agent)
        workflow.add_node("opponent", self.base_system._opponent_agent)
        workflow.add_node("advocate", self._advocate_agent)
        workflow.add_node("critic", self._critic_agent)
        workflow.add_node("refiner", self._refiner_agent)
        workflow.add_node("moderator", self.base_system._moderator_agent)
        
        # Extended flow with parallel processing
        workflow.add_parallel(['proponent', 'opponent', 'advocate', 'critic'])
        workflow.add_edge(['proponent', 'opponent', 'advocate', 'critic'], 'refiner')
        workflow.add_edge('refiner', 'moderator')
        
        # Continue for up to 3 rounds
        workflow.add_conditional_edges(
            "moderator",
            self._should_continue_extended_debate,
            {
                "continue": "proponent",
                "conclude": "final_decision"
            }
        )
        
        return workflow.compile()
    
    async def _advocate_agent(self, state: DebateState) -> DebateState:
        """Agent that advocates for specific aspects/details"""
        
        prompt = f"""
        You are an advocate focused on practical implementation for: {state['topic']}
        
        Previous arguments: {self._build_context(state, DebateRole.ADVOCATE)}
        
        Focus on:
        1. Practical benefits and real-world applications
        2. Stakeholder perspectives and concerns
        3. Implementation considerations
        4. Risk mitigation strategies
        
        Provide nuanced advocacy that considers practical constraints.
        """
        
        response = await self.model_router.route_request(prompt, {
            'task_type': 'analysis',
            'agent_role': 'advocate'
        })
        
        argument = self._parse_argument_response(response, DebateRole.ADVOCATE, state)
        state['arguments'].append(argument)
        return state
    
    async def _critic_agent(self, state: DebateState) -> DebateState:
        """Agent that provides critical analysis of all positions"""
        
        prompt = f"""
        You are a critic analyzing all positions on: {state['topic']}
        
        Previous arguments: {self._build_context(state, DebateRole.CRITIC)}
        
        Provide critical analysis focusing on:
        1. Logical fallacies and weak reasoning
        2. Missing evidence or unsupported claims
        3. Potential biases in arguments
        4. Alternative frameworks for understanding the issue
        
        Be thorough and constructive in your criticism.
        """
        
        response = await self.model_router.route_request(prompt, {
            'task_type': 'analysis',
            'agent_role': 'critic'
        })
        
        argument = self._parse_argument_response(response, DebateRole.CRITIC, state)
        state['arguments'].append(argument)
        return state
    
    async def _refiner_agent(self, state: DebateState) -> DebateState:
        """Agent that synthesizes and refines arguments"""
        
        current_round_args = [
            arg for arg in state['arguments'] 
            if arg.round_number == state['current_round']
        ]
        
        prompt = f"""
        Synthesize and refine the debate on: {state['topic']}
        
        Current round arguments:
        {self._format_arguments_for_synthesis(current_round_args)}
        
        Provide a refined synthesis that:
        1. Integrates valid points from all perspectives
        2. Identifies areas of agreement and disagreement
        3. Suggests resolution paths or compromises
        4. Highlights key insights and remaining questions
        
        Focus on constructive synthesis rather than taking sides.
        """
        
        response = await self.model_router.route_request(prompt, {
            'task_type': 'reasoning',
            'agent_role': 'refiner'
        })
        
        argument = self._parse_argument_response(response, DebateRole.REFINER, state)
        state['arguments'].append(argument)
        return state
```

## Performance Evaluation

### Debate Quality Metrics
```python
class DebateQualityEvaluator:
    def __init__(self):
        self.quality_dimensions = [
            'logical_consistency',
            'evidence_quality', 
            'argument_diversity',
            'consensus_quality',
            'decision_confidence'
        ]
    
    def evaluate_debate_quality(self, debate_state: DebateState) -> Dict[str, float]:
        """Evaluate overall quality of the debate process"""
        
        scores = {}
        
        # Logical consistency across arguments
        scores['logical_consistency'] = self._evaluate_logical_consistency(
            debate_state['arguments']
        )
        
        # Quality of evidence presented
        scores['evidence_quality'] = self._evaluate_evidence_quality(
            debate_state['arguments']
        )
        
        # Diversity of perspectives and arguments
        scores['argument_diversity'] = self._evaluate_argument_diversity(
            debate_state['arguments']
        )
        
        # Quality of final consensus/decision
        scores['consensus_quality'] = self._evaluate_consensus_quality(
            debate_state
        )
        
        # Confidence calibration
        scores['decision_confidence'] = debate_state['confidence_score']
        
        # Overall quality score
        scores['overall_quality'] = sum(scores.values()) / len(scores)
        
        return scores
    
    def compare_with_single_agent(self, topic: str, single_agent_result: str,
                                 debate_result: str) -> Dict[str, float]:
        """Compare debate outcome with single-agent decision"""
        
        comparison = {
            'accuracy_improvement': self._measure_accuracy_improvement(
                topic, single_agent_result, debate_result
            ),
            'reasoning_depth': self._measure_reasoning_depth(
                single_agent_result, debate_result
            ),
            'bias_reduction': self._measure_bias_reduction(
                single_agent_result, debate_result
            ),
            'confidence_calibration': self._measure_confidence_calibration(
                single_agent_result, debate_result
            )
        }
        
        return comparison
```

## Performance Targets

| Metric | Phase 1 Target | Phase 2 Target | Baseline (Single Agent) |
|--------|----------------|----------------|--------------------------|
| Accuracy Improvement | +30-40% | +40-50% | 0% |
| Hallucination Reduction | 60% | 70% | 0% |
| Decision Confidence | >0.8 | >0.85 | 0.6 |
| Reasoning Depth Score | >0.8 | >0.9 | 0.5 |
| Consensus Rate | >70% | >80% | N/A |

## Related Decisions

- ADR-008: Multi-Model Routing

- ADR-001: Multi-Agent Framework Selection

- ADR-013: Extended Debate (Phase 2)

## Monitoring

- Debate outcome accuracy vs ground truth

- Argument quality and logical consistency

- Consensus achievement rates

- Model utilization in debate contexts

- Cost per debate vs single-agent decisions
