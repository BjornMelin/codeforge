# ADR-013: Extended Debate (Phase 2)

**Status**: Proposed (for Phase 2)  

**Date**: 2025-07-20  

**Deciders**: CodeForge AI Team  

## Context

Scale Phase 1 3-agent debate for toughest tasks. Phase 1's 3-agent debate (pro/con/moderator) works well for standard decisions but needs scaling for the most complex reasoning, planning, and architectural decisions requiring deeper analysis.

## Problem Statement

Phase 1 balanced approach needs enhanced depth for complex tasks. Requirements include:

- Deeper analysis for complex architectural decisions

- More nuanced reasoning for multi-faceted problems

- Enhanced perspective diversity for comprehensive evaluation

- Scalable complexity without overwhelming simple tasks

- Maintain cost efficiency for routine decisions

## Decision

**Toggle to 5 agents** (add advocate/critic/refiner) + 3 rounds for +10% gains in complex reasoning tasks.

## Alternatives Considered

| Approach | Pros | Cons | Score |
|----------|------|------|-------|
| **5-Agent Toggle System** | Scalable complexity, optimal for hard problems | Higher resource usage, complexity | **8.2** |
| Keep 3-agent system | Sufficient for most tasks, proven approach | Limited depth for complex problems | 8.0 |
| Always use 5 agents | Maximum depth for all tasks | Overkill for simple decisions, high costs | 7.5 |
| 7+ agent system | Extreme depth potential | Diminishing returns, coordination complexity | 7.0 |

## Rationale

- **Scalable gains (8.2)**: +10% improvement on complex tasks while preserving efficiency

- **Smart resource allocation**: Only uses extended debate when complexity justifies it

- **Proven extension**: Natural progression from established 3-agent foundation

- **Cost-conscious**: Toggle prevents unnecessary overhead on routine tasks

## Consequences

### Positive

- Enhanced reasoning quality for the most challenging problems

- Better architectural and design decisions through diverse perspectives

- Improved handling of multi-stakeholder considerations

- Scalable complexity based on task requirements

### Negative

- Higher computational costs and latency for complex tasks

- Increased coordination complexity with more agents

- Risk of analysis paralysis on edge cases

### Neutral

- Higher load toggle mechanism for selective use

- Integration with existing debate infrastructure

## Implementation Notes

### Extended Debate Architecture
```python
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

class ExtendedDebateRole(Enum):
    PROPONENT = "proponent"        # Phase 1
    OPPONENT = "opponent"          # Phase 1
    MODERATOR = "moderator"        # Phase 1
    ADVOCATE = "advocate"          # Phase 2
    CRITIC = "critic"              # Phase 2
    REFINER = "refiner"            # Phase 2

@dataclass
class ComplexityAssessment:
    score: float  # 0.0-1.0
    factors: List[str]
    reasoning: str
    recommended_agents: int
    recommended_rounds: int

class ExtendedDebateSystem:
    def __init__(self, base_debate_system, model_router):
        self.base_system = base_debate_system
        self.model_router = model_router
        self.complexity_threshold = 0.7  # Use 5 agents for complexity > 0.7
        
        # Extended agent configurations
        self.agent_configs = {
            ExtendedDebateRole.ADVOCATE: {
                'focus': 'practical_benefits',
                'perspective': 'implementation_focused',
                'prompt_style': 'supportive_practical'
            },
            ExtendedDebateRole.CRITIC: {
                'focus': 'potential_problems',
                'perspective': 'risk_assessment',
                'prompt_style': 'analytical_cautious'
            },
            ExtendedDebateRole.REFINER: {
                'focus': 'synthesis_optimization',
                'perspective': 'solution_oriented',
                'prompt_style': 'constructive_integrative'
            }
        }
    
    async def assess_complexity(self, topic: str, context: Dict[str, Any]) -> ComplexityAssessment:
        """Assess whether extended debate is warranted"""
        
        complexity_factors = []
        base_score = 0.3
        
        # Topic length and detail
        if len(topic) > 200:
            base_score += 0.1
            complexity_factors.append("detailed_topic")
        
        # Context complexity indicators
        if context.get('stakeholder_count', 0) > 3:
            base_score += 0.15
            complexity_factors.append("multiple_stakeholders")
        
        if context.get('technical_domains', 0) > 2:
            base_score += 0.15
            complexity_factors.append("cross_domain_complexity")
        
        if context.get('time_horizon') == 'long_term':
            base_score += 0.1
            complexity_factors.append("long_term_implications")
        
        if context.get('budget_impact') == 'high':
            base_score += 0.1
            complexity_factors.append("high_impact_decision")
        
        # AI-based complexity assessment
        ai_assessment = await self._ai_complexity_assessment(topic, context)
        base_score += ai_assessment * 0.2
        
        if ai_assessment > 0.7:
            complexity_factors.append("ai_assessed_high_complexity")
        
        # Determine agent count and rounds
        if base_score >= 0.8:
            recommended_agents = 5
            recommended_rounds = 3
        elif base_score >= 0.7:
            recommended_agents = 5
            recommended_rounds = 2
        else:
            recommended_agents = 3
            recommended_rounds = 2
        
        return ComplexityAssessment(
            score=min(base_score, 1.0),
            factors=complexity_factors,
            reasoning=f"Complexity score: {base_score:.2f}, factors: {', '.join(complexity_factors)}",
            recommended_agents=recommended_agents,
            recommended_rounds=recommended_rounds
        )
    
    async def _ai_complexity_assessment(self, topic: str, context: Dict[str, Any]) -> float:
        """Use AI to assess topic complexity"""
        
        prompt = f"""
        Assess the complexity of this decision/topic on a scale of 0.0-1.0:
        
        Topic: {topic}
        Context: {context}
        
        Consider:
        - Number of variables and dependencies
        - Potential for unintended consequences
        - Stakeholder diversity and conflicting interests
        - Technical complexity and domain expertise required
        - Long-term vs short-term trade-offs
        
        Return only a number between 0.0 and 1.0.
        """
        
        response = await self.model_router.route_request(
            prompt, {'task_type': 'analysis', 'agent_role': 'complexity_assessor'}
        )
        
        try:
            return float(response.strip())
        except:
            return 0.5  # Default moderate complexity
```

### Extended Agent Implementations
```python
class ExtendedDebateAgents:
    def __init__(self, model_router):
        self.model_router = model_router
    
    async def advocate_agent(self, state: DebateState) -> DebateState:
        """Agent focused on practical advocacy and implementation benefits"""
        
        context = self._build_extended_context(state, ExtendedDebateRole.ADVOCATE)
        
        prompt = f"""
        You are an ADVOCATE for practical implementation of: {state['topic']}
        
        Previous debate rounds:
        {context}
        
        Focus on PRACTICAL BENEFITS and IMPLEMENTATION ADVANTAGES:
        
        1. Real-world applicability and usefulness
        2. Stakeholder benefits and value propositions
        3. Implementation feasibility and pathways
        4. Resource utilization and efficiency gains
        5. Risk mitigation through proper implementation
        
        Provide strong advocacy that considers:
        - Who benefits and how
        - What makes this practically valuable
        - How implementation risks can be managed
        - Why the benefits outweigh the costs
        
        Be concrete and implementation-focused in your advocacy.
        """
        
        response = await self.model_router.route_request(
            prompt, {'task_type': 'reasoning', 'agent_role': 'advocate'}
        )
        
        argument = self._parse_extended_argument(response, ExtendedDebateRole.ADVOCATE, state)
        state['arguments'].append(argument)
        
        return state
    
    async def critic_agent(self, state: DebateState) -> DebateState:
        """Agent focused on critical analysis and risk assessment"""
        
        context = self._build_extended_context(state, ExtendedDebateRole.CRITIC)
        
        prompt = f"""
        You are a CRITIC providing rigorous analysis of: {state['topic']}
        
        Previous debate rounds:
        {context}
        
        Focus on CRITICAL ANALYSIS and RISK ASSESSMENT:
        
        1. Logical fallacies and weak reasoning in arguments
        2. Unstated assumptions and their validity
        3. Potential negative consequences and risks
        4. Resource constraints and implementation challenges
        5. Alternative perspectives and frameworks
        
        Provide constructive criticism that examines:
        - What could go wrong and why
        - Where the logic breaks down
        - What important factors are being overlooked
        - How the costs might outweigh benefits
        
        Be thorough and analytical, but constructive in your criticism.
        """
        
        response = await self.model_router.route_request(
            prompt, {'task_type': 'analysis', 'agent_role': 'critic'}
        )
        
        argument = self._parse_extended_argument(response, ExtendedDebateRole.CRITIC, state)
        state['arguments'].append(argument)
        
        return state
    
    async def refiner_agent(self, state: DebateState) -> DebateState:
        """Agent focused on synthesis and solution refinement"""
        
        current_round_args = [
            arg for arg in state['arguments']
            if arg.round_number == state['current_round']
        ]
        
        prompt = f"""
        You are a REFINER synthesizing insights on: {state['topic']}
        
        Current round arguments from all perspectives:
        {self._format_arguments_for_synthesis(current_round_args)}
        
        Focus on SYNTHESIS and SOLUTION REFINEMENT:
        
        1. Integrate valid points from all perspectives
        2. Identify areas of convergence and divergence
        3. Propose refined approaches that address concerns
        4. Suggest compromise solutions or hybrid approaches
        5. Highlight key insights and remaining questions
        
        Provide constructive synthesis that:
        - Builds bridges between opposing viewpoints
        - Suggests practical compromises or alternatives
        - Identifies the strongest elements from each position
        - Proposes next steps for resolution
        
        Focus on constructive integration rather than taking sides.
        """
        
        response = await self.model_router.route_request(
            prompt, {'task_type': 'reasoning', 'agent_role': 'refiner'}
        )
        
        argument = self._parse_extended_argument(response, ExtendedDebateRole.REFINER, state)
        state['arguments'].append(argument)
        
        return state
```

### Dynamic Workflow Configuration
```python
class DynamicDebateWorkflow:
    def __init__(self, base_debate, extended_agents):
        self.base_debate = base_debate
        self.extended_agents = extended_agents
    
    def create_adaptive_workflow(self, complexity_assessment: ComplexityAssessment) -> StateGraph:
        """Create workflow adapted to complexity requirements"""
        
        workflow = StateGraph(DebateState)
        
        if complexity_assessment.recommended_agents == 5:
            # Extended 5-agent workflow
            workflow = self._create_extended_workflow(complexity_assessment.recommended_rounds)
        else:
            # Standard 3-agent workflow  
            workflow = self.base_debate.create_debate_graph()
        
        return workflow
    
    def _create_extended_workflow(self, max_rounds: int) -> StateGraph:
        """Create 5-agent extended debate workflow"""
        
        workflow = StateGraph(DebateState)
        
        # Phase 1: Parallel initial positions
        workflow.add_node("proponent", self.base_debate._proponent_agent)
        workflow.add_node("opponent", self.base_debate._opponent_agent)
        workflow.add_node("advocate", self.extended_agents.advocate_agent)
        workflow.add_node("critic", self.extended_agents.critic_agent)
        
        # Phase 2: Synthesis
        workflow.add_node("refiner", self.extended_agents.refiner_agent)
        
        # Phase 3: Moderation
        workflow.add_node("moderator", self.base_debate._moderator_agent)
        
        # Phase 4: Decision
        workflow.add_node("final_decision", self.base_debate._finalize_decision)
        
        # Workflow edges
        workflow.add_parallel(['proponent', 'opponent', 'advocate', 'critic'])
        workflow.add_edge(['proponent', 'opponent', 'advocate', 'critic'], 'refiner')
        workflow.add_edge('refiner', 'moderator')
        
        # Conditional continuation for multiple rounds
        workflow.add_conditional_edges(
            'moderator',
            lambda state: self._should_continue_extended_debate(state, max_rounds),
            {
                'continue': 'proponent',
                'conclude': 'final_decision'
            }
        )
        
        workflow.set_entry_point('proponent')
        
        return workflow.compile()
    
    def _should_continue_extended_debate(self, state: DebateState, max_rounds: int) -> str:
        """Determine if extended debate should continue"""
        
        if state['current_round'] >= max_rounds:
            return 'conclude'
        
        if state['consensus_reached']:
            return 'conclude'
        
        # Check for diminishing returns
        if state['current_round'] >= 2:
            recent_quality = self._assess_recent_argument_quality(state)
            if recent_quality < 0.6:  # Arguments not adding much value
                return 'conclude'
        
        return 'continue'
```

### Performance Analysis and Optimization
```python
class ExtendedDebateAnalyzer:
    def __init__(self):
        self.quality_metrics = {
            'argument_diversity': 0.0,
            'reasoning_depth': 0.0,
            'synthesis_quality': 0.0,
            'decision_confidence': 0.0
        }
    
    def analyze_extended_debate(self, debate_state: DebateState) -> Dict[str, float]:
        """Analyze the quality and effectiveness of extended debate"""
        
        analysis = {}
        
        # Argument diversity - how many different perspectives were covered
        analysis['argument_diversity'] = self._calculate_argument_diversity(
            debate_state['arguments']
        )
        
        # Reasoning depth - quality of analysis and evidence
        analysis['reasoning_depth'] = self._calculate_reasoning_depth(
            debate_state['arguments']
        )
        
        # Synthesis quality - how well different views were integrated
        synthesis_args = [arg for arg in debate_state['arguments'] 
                         if arg.role == ExtendedDebateRole.REFINER]
        analysis['synthesis_quality'] = self._evaluate_synthesis_quality(synthesis_args)
        
        # Decision confidence and justification
        analysis['decision_confidence'] = debate_state.get('confidence_score', 0.5)
        
        # Cost-benefit analysis
        analysis['cost_efficiency'] = self._calculate_cost_efficiency(
            debate_state, analysis
        )
        
        return analysis
    
    def _calculate_argument_diversity(self, arguments: List[DebateArgument]) -> float:
        """Calculate diversity of perspectives and argument types"""
        
        if not arguments:
            return 0.0
        
        # Count unique themes and perspectives
        unique_themes = set()
        for arg in arguments:
            # Extract key themes (simplified)
            themes = self._extract_argument_themes(arg.content)
            unique_themes.update(themes)
        
        # Normalize by expected maximum diversity
        max_expected_themes = 15  # Reasonable upper bound
        diversity_score = min(len(unique_themes) / max_expected_themes, 1.0)
        
        return diversity_score
    
    def _calculate_reasoning_depth(self, arguments: List[DebateArgument]) -> float:
        """Assess the depth and quality of reasoning"""
        
        if not arguments:
            return 0.0
        
        depth_scores = []
        
        for arg in arguments:
            # Assess various depth indicators
            evidence_score = len(arg.supporting_evidence) / 5.0  # Normalize by expected max
            length_score = min(len(arg.content) / 1000, 1.0)  # Longer tends to be deeper
            confidence_score = arg.confidence
            
            # Combined depth score
            depth = (evidence_score * 0.4 + length_score * 0.3 + confidence_score * 0.3)
            depth_scores.append(min(depth, 1.0))
        
        return sum(depth_scores) / len(depth_scores)
    
    def compare_with_standard_debate(self, extended_result: Dict, 
                                   standard_result: Dict) -> Dict[str, float]:
        """Compare extended vs standard debate outcomes"""
        
        comparison = {
            'quality_improvement': (
                extended_result['reasoning_depth'] - 
                standard_result.get('reasoning_depth', 0.5)
            ),
            'diversity_gain': (
                extended_result['argument_diversity'] - 
                standard_result.get('argument_diversity', 0.5)
            ),
            'confidence_improvement': (
                extended_result['decision_confidence'] - 
                standard_result.get('decision_confidence', 0.5)
            ),
            'cost_multiplier': extended_result.get('total_cost', 0) / max(
                standard_result.get('total_cost', 1), 1
            )
        }
        
        return comparison
```

## Performance Targets

| Metric | 3-Agent Baseline | 5-Agent Target | Improvement |
|--------|------------------|----------------|-------------|
| Reasoning Quality | 0.7 | 0.8 | +14% |
| Decision Confidence | 0.75 | 0.85 | +13% |
| Argument Diversity | 0.6 | 0.8 | +33% |
| Complex Task Success | 70% | 80% | +10% |
| Cost per Decision | $0.50 | $1.25 | 2.5x |

## Usage Guidelines

### When to Use Extended Debate

- **Architecture decisions** with multiple technical approaches

- **Resource allocation** decisions affecting multiple teams

- **Strategic planning** with long-term implications

- **Cross-functional initiatives** requiring diverse expertise

- **High-stakes decisions** with significant downside risk

### When to Use Standard Debate

- **Implementation details** within established architecture

- **Routine operational decisions** with clear precedent

- **Simple feature specifications** with limited scope

- **Quick technical choices** with reversible outcomes

## Cost Management

### Smart Triggering
```python
class DebateCostManager:
    def __init__(self):
        self.daily_budget = 50.0  # $50/day for all debates
        self.extended_debate_cost = 1.25  # $1.25 per extended debate
        self.standard_debate_cost = 0.50  # $0.50 per standard debate
        self.current_usage = 0.0
    
    def should_use_extended_debate(self, complexity_score: float, 
                                 topic_importance: str) -> bool:
        """Cost-aware decision on debate type"""
        
        # Check budget
        if self.current_usage + self.extended_debate_cost > self.daily_budget:
            return False
        
        # High importance always gets extended debate
        if topic_importance == 'critical':
            return True
        
        # Use complexity threshold
        if complexity_score > 0.8:
            return True
        elif complexity_score > 0.7 and topic_importance == 'high':
            return True
        
        return False
```

## Related Decisions

- ADR-009: Debate Agents (Phase 1)

- ADR-008: Multi-Model Routing

- ADR-001: Multi-Agent Framework Selection

## Monitoring

- Extended vs standard debate quality comparisons

- Cost-benefit analysis per decision type

- User satisfaction with complex decision outcomes

- Agent coordination efficiency in 5-agent mode

- Optimal complexity threshold calibration
