# ADR-014: Federated Basics (Phase 2)

**Status**: Proposed (for Phase 2)  

**Date**: 2025-07-20  

**Deciders**: CodeForge AI Team  

## Context

Privacy in Phase 2 multi-agent training. As the system scales to multiple organizations and privacy-sensitive environments, there's a need for basic federated learning capabilities that allow model improvement without centralizing sensitive data.

## Problem Statement

Phase 1 centralized approach needs local aggregation for privacy. Requirements include:

- Privacy-preserving model updates across distributed deployments

- Local data retention with collaborative learning benefits

- Basic model aggregation without exposing proprietary information

- Simple implementation that doesn't compromise Phase 1 reliability

- Optional federation for organizations that require data locality

## Decision

**Flower lib for basic model aggregation** enabling privacy-focused multi-agent personalization without central data exposure.

## Alternatives Considered

| Approach | Pros | Cons | Score |
|----------|------|------|-------|
| **Flower Framework** | Proven framework, privacy-focused, simple setup | Additional dependency, basic features only | **8.0** |
| No federated learning | Simpler, no privacy complexity | Misses collaboration benefits, privacy limitations | 7.0 |
| Custom federated system | Full control, optimized | High development cost, security complexity | 7.2 |
| PySyft integration | Advanced privacy features | Complex setup, heavyweight dependency | 7.5 |

## Rationale

- **Simple privacy gains (8.0)**: Basic but effective privacy preservation

- **Proven framework**: Flower is battle-tested for federated learning

- **Optional adoption**: Organizations can choose federated or centralized

- **Foundation building**: Establishes patterns for advanced federation later

## Consequences

### Positive

- Enhanced privacy for sensitive data environments

- Collaborative learning across organizations without data sharing

- Maintained local control over proprietary information

- Foundation for advanced federated capabilities

### Negative

- Additional system complexity and dependencies

- Network coordination overhead between nodes

- Limited to basic aggregation algorithms initially

### Neutral

- Local data processing only

- Optional adoption based on privacy requirements

## Implementation Notes

### Federated Architecture Overview
```python
import flwr as fl
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import json
import asyncio

@dataclass
class FederationConfig:
    node_id: str
    federation_name: str
    aggregation_strategy: str = "FedAvg"
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    privacy_budget: float = 1.0

class CodeForgeFederatedClient(fl.client.NumPyClient):
    """Federated learning client for CodeForge AI"""
    
    def __init__(self, node_id: str, local_agent_system):
        self.node_id = node_id
        self.local_system = local_agent_system
        self.privacy_filter = PrivacyFilter()
        
        # Model components that can be federated
        self.federatable_components = {
            'task_routing_weights': self._get_routing_weights,
            'debate_effectiveness_scores': self._get_debate_scores,
            'performance_patterns': self._get_performance_patterns
        }
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Extract model parameters for federation"""
        
        parameters = []
        
        for component_name in config.get('components', self.federatable_components.keys()):
            if component_name in self.federatable_components:
                component_params = self.federatable_components[component_name]()
                
                # Apply privacy filtering
                filtered_params = self.privacy_filter.filter_parameters(
                    component_params, component_name
                )
                
                parameters.extend(filtered_params)
        
        return parameters
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update local model with federated parameters"""
        
        # Apply federated updates to local system
        param_index = 0
        
        for component_name in self.federatable_components.keys():
            component_size = self._get_component_size(component_name)
            
            if param_index + component_size <= len(parameters):
                component_params = parameters[param_index:param_index + component_size]
                self._update_component_parameters(component_name, component_params)
                param_index += component_size
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train local model and return updated parameters"""
        
        # Update local model with global parameters
        self.set_parameters(parameters)
        
        # Perform local training/optimization
        local_improvements = self._perform_local_optimization(config)
        
        # Get updated parameters
        updated_params = self.get_parameters(config)
        
        # Return parameters and metrics
        return updated_params, local_improvements['sample_count'], local_improvements['metrics']
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """Evaluate model performance"""
        
        self.set_parameters(parameters)
        
        # Evaluate local performance
        evaluation_results = self._evaluate_local_performance(config)
        
        return (
            evaluation_results['loss'],
            evaluation_results['sample_count'], 
            evaluation_results['metrics']
        )
```

### Privacy Filtering System
```python
class PrivacyFilter:
    """Ensures sensitive information is not shared in federation"""
    
    def __init__(self):
        self.sensitive_patterns = [
            r'api_key', r'password', r'token', r'secret',
            r'user.*data', r'proprietary', r'confidential'
        ]
        
        self.aggregation_methods = {
            'differential_privacy': self._apply_differential_privacy,
            'secure_aggregation': self._apply_secure_aggregation,
            'noise_injection': self._apply_noise_injection
        }
    
    def filter_parameters(self, parameters: Dict[str, Any], 
                         component_name: str) -> List[np.ndarray]:
        """Filter parameters to remove sensitive information"""
        
        filtered_params = []
        
        for param_name, param_value in parameters.items():
            # Check if parameter contains sensitive information
            if self._is_sensitive_parameter(param_name, param_value):
                continue
            
            # Convert to numpy array if needed
            if isinstance(param_value, (int, float)):
                param_array = np.array([param_value])
            elif isinstance(param_value, (list, tuple)):
                param_array = np.array(param_value)
            else:
                param_array = np.array([0.0])  # Safe default
            
            # Apply privacy protection
            protected_param = self._apply_privacy_protection(
                param_array, component_name
            )
            
            filtered_params.append(protected_param)
        
        return filtered_params
    
    def _is_sensitive_parameter(self, param_name: str, param_value: Any) -> bool:
        """Check if parameter contains sensitive information"""
        
        param_name_lower = param_name.lower()
        
        # Check against sensitive patterns
        for pattern in self.sensitive_patterns:
            if re.search(pattern, param_name_lower):
                return True
        
        # Check value content for sensitive data
        if isinstance(param_value, str):
            for pattern in self.sensitive_patterns:
                if re.search(pattern, param_value.lower()):
                    return True
        
        return False
    
    def _apply_privacy_protection(self, param_array: np.ndarray, 
                                component_name: str) -> np.ndarray:
        """Apply privacy protection to parameter array"""
        
        # Use differential privacy for performance metrics
        if 'performance' in component_name:
            return self._apply_differential_privacy(param_array)
        
        # Use noise injection for routing weights
        elif 'routing' in component_name:
            return self._apply_noise_injection(param_array, noise_scale=0.01)
        
        # Default: minimal noise injection
        else:
            return self._apply_noise_injection(param_array, noise_scale=0.005)
    
    def _apply_differential_privacy(self, data: np.ndarray, 
                                  epsilon: float = 1.0) -> np.ndarray:
        """Apply differential privacy with Laplace noise"""
        
        sensitivity = np.max(data) - np.min(data)
        if sensitivity == 0:
            sensitivity = 1.0
        
        noise_scale = sensitivity / epsilon
        noise = np.random.laplace(0, noise_scale, data.shape)
        
        return data + noise
    
    def _apply_noise_injection(self, data: np.ndarray, 
                             noise_scale: float = 0.01) -> np.ndarray:
        """Apply Gaussian noise for basic privacy"""
        
        noise = np.random.normal(0, noise_scale * np.std(data), data.shape)
        return data + noise
```

### Federated Coordination Server
```python
class CodeForgeFederationServer:
    """Coordination server for federated learning"""
    
    def __init__(self, config: FederationConfig):
        self.config = config
        self.client_registry: Dict[str, Dict] = {}
        self.aggregation_history: List[Dict] = []
        
    def create_strategy(self) -> fl.server.strategy.Strategy:
        """Create federated learning strategy"""
        
        if self.config.aggregation_strategy == "FedAvg":
            return fl.server.strategy.FedAvg(
                min_fit_clients=self.config.min_fit_clients,
                min_evaluate_clients=self.config.min_evaluate_clients,
                min_available_clients=self.config.min_available_clients,
                on_fit_config_fn=self._get_fit_config,
                on_evaluate_config_fn=self._get_evaluate_config,
                evaluate_metrics_aggregation_fn=self._aggregate_metrics
            )
        else:
            # Can add other strategies (FedProx, FedOpt, etc.)
            return fl.server.strategy.FedAvg()
    
    def _get_fit_config(self, server_round: int) -> Dict[str, Any]:
        """Configure training round"""
        return {
            "server_round": server_round,
            "local_epochs": 1,
            "components": ["task_routing_weights", "performance_patterns"]
        }
    
    def _get_evaluate_config(self, server_round: int) -> Dict[str, Any]:
        """Configure evaluation round"""
        return {
            "server_round": server_round,
            "components": ["task_routing_weights", "performance_patterns"]
        }
    
    def _aggregate_metrics(self, metrics: List[Tuple[int, Dict]]) -> Dict:
        """Aggregate evaluation metrics from clients"""
        
        aggregated = {
            "accuracy": 0.0,
            "efficiency": 0.0,
            "total_samples": 0
        }
        
        total_samples = sum(num_samples for num_samples, _ in metrics)
        
        for num_samples, client_metrics in metrics:
            weight = num_samples / total_samples
            
            aggregated["accuracy"] += weight * client_metrics.get("accuracy", 0.0)
            aggregated["efficiency"] += weight * client_metrics.get("efficiency", 0.0)
        
        aggregated["total_samples"] = total_samples
        
        return aggregated
    
    async def start_federation_round(self) -> Dict[str, Any]:
        """Start a federation training round"""
        
        try:
            # Start Flower server
            strategy = self.create_strategy()
            
            history = fl.server.start_server(
                server_address="[::]:8080",
                config=fl.server.ServerConfig(num_rounds=1),
                strategy=strategy
            )
            
            # Record aggregation results
            round_results = {
                "timestamp": time.time(),
                "participants": len(self.client_registry),
                "history": history,
                "success": True
            }
            
            self.aggregation_history.append(round_results)
            
            return round_results
            
        except Exception as e:
            logger.error(f"Federation round failed: {e}")
            return {"success": False, "error": str(e)}
```

### Local System Integration
```python
class FederatedCodeForgeSystem:
    """Integration layer for federated capabilities"""
    
    def __init__(self, local_system, federation_config: Optional[FederationConfig] = None):
        self.local_system = local_system
        self.federation_config = federation_config
        self.federated_client = None
        self.federation_enabled = federation_config is not None
        
        if self.federation_enabled:
            self.federated_client = CodeForgeFederatedClient(
                federation_config.node_id, local_system
            )
    
    async def enable_federation(self, config: FederationConfig):
        """Enable federated learning capabilities"""
        
        self.federation_config = config
        self.federated_client = CodeForgeFederatedClient(
            config.node_id, self.local_system
        )
        self.federation_enabled = True
        
        logger.info(f"Federation enabled for node {config.node_id}")
    
    async def participate_in_federation(self) -> bool:
        """Participate in a federation round"""
        
        if not self.federation_enabled:
            logger.warning("Federation not enabled")
            return False
        
        try:
            # Connect to federation server
            fl.client.start_numpy_client(
                server_address="localhost:8080",
                client=self.federated_client
            )
            
            logger.info("Successfully participated in federation round")
            return True
            
        except Exception as e:
            logger.error(f"Federation participation failed: {e}")
            return False
    
    async def get_federation_benefits(self) -> Dict[str, float]:
        """Analyze benefits gained from federation"""
        
        if not self.federation_enabled:
            return {"enabled": False}
        
        # Compare performance before and after federation
        benefits = {
            "routing_accuracy_improvement": self._measure_routing_improvement(),
            "debate_quality_improvement": self._measure_debate_improvement(),
            "overall_efficiency_gain": self._measure_efficiency_gain(),
            "privacy_score": self._calculate_privacy_score()
        }
        
        return benefits
    
    def _measure_routing_improvement(self) -> float:
        """Measure improvement in model routing accuracy"""
        # Implementation would compare routing decisions before/after federation
        return 0.05  # Placeholder: 5% improvement
    
    def _measure_debate_improvement(self) -> float:
        """Measure improvement in debate quality"""
        # Implementation would analyze debate outcomes before/after federation
        return 0.03  # Placeholder: 3% improvement
    
    def _measure_efficiency_gain(self) -> float:
        """Measure overall system efficiency improvement"""
        # Implementation would measure task completion rates, etc.
        return 0.08  # Placeholder: 8% improvement
    
    def _calculate_privacy_score(self) -> float:
        """Calculate privacy preservation score"""
        # Implementation would assess data exposure risks
        return 0.95  # Placeholder: 95% privacy preservation
```

### Use Case Examples
```python
class FederationUseCases:
    """Example implementations for common federation scenarios"""
    
    @staticmethod
    async def cross_organization_learning():
        """Enable learning across organizations without data sharing"""
        
        # Organization A
        org_a_config = FederationConfig(
            node_id="org_a",
            federation_name="codeforge_consortium",
            min_fit_clients=2
        )
        
        org_a_system = FederatedCodeForgeSystem(
            local_system=CodeForgeSystem(),
            federation_config=org_a_config
        )
        
        # Organization B
        org_b_config = FederationConfig(
            node_id="org_b", 
            federation_name="codeforge_consortium",
            min_fit_clients=2
        )
        
        org_b_system = FederatedCodeForgeSystem(
            local_system=CodeForgeSystem(),
            federation_config=org_b_config
        )
        
        # Both participate in learning without sharing data
        await asyncio.gather(
            org_a_system.participate_in_federation(),
            org_b_system.participate_in_federation()
        )
    
    @staticmethod
    async def privacy_sensitive_deployment():
        """Deploy in privacy-sensitive environment with federation"""
        
        config = FederationConfig(
            node_id="healthcare_org",
            federation_name="healthcare_ai_consortium",
            privacy_budget=0.5  # Strict privacy budget
        )
        
        system = FederatedCodeForgeSystem(
            local_system=CodeForgeSystem(),
            federation_config=config
        )
        
        # Process sensitive healthcare data locally
        # while benefiting from collaborative learning
        await system.participate_in_federation()
        
        benefits = await system.get_federation_benefits()
        logger.info(f"Privacy-preserving benefits: {benefits}")
```

## Performance Targets

| Metric | Local Only | Federated Target | Improvement |
|--------|------------|------------------|-------------|
| Routing Accuracy | 85% | 90% | +5% |
| Debate Quality | 80% | 85% | +6% |
| Privacy Score | 70% | 95% | +36% |
| System Efficiency | 75% | 80% | +7% |
| Data Locality | 0% | 100% | Full local control |

## Security Considerations

### Privacy Protection

- Differential privacy for sensitive metrics

- Secure aggregation protocols

- Local data processing only

- Encrypted communication channels

### Attack Mitigation

- Model poisoning detection

- Gradient inversion protection

- Membership inference defense

- Byzantine fault tolerance

## Deployment Options

### Consortium Mode

- Multiple organizations collaborate

- Shared federation server

- Common governance model

- Standardized privacy policies

### Hybrid Mode

- Optional federation participation

- Local-first with federation benefits

- Gradual privacy budget adjustment

- Fallback to centralized when needed

## Related Decisions

- ADR-005: Caching and Shared Context Layer

- ADR-010: Task Management System

- ADR-002: Database and Memory System

## Monitoring

- Federation participation rates and success

- Privacy budget utilization and effectiveness

- Collaborative learning benefit measurements

- Network coordination overhead tracking

- Security incident detection and response
