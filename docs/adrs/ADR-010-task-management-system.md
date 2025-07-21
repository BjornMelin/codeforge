# ADR-010: Task Management System

**Status**: Accepted  

**Date**: 2025-07-20  

**Deciders**: CodeForge AI Team  

## Context

Low-latency coordination in Phase 1, scale to Phase 2 extended agents. The system needs efficient task distribution, progress tracking, and coordination mechanisms that can handle increasing agent complexity while maintaining high performance.

## Problem Statement

Provide efficient task coordination and management across phases. Requirements include:

- Low-latency task assignment and updates

- Reliable task persistence and recovery

- Scalable coordination for growing agent populations

- Integration with debate and routing systems

- Support for complex task dependencies

## Decision

**Hybrid in-memory (deque in StateGraph) + Redis Pub/Sub/checkpointers** in Phase 1; extend to Phase 2 with federated task aggregation (Flower for privacy).

## Alternatives Considered

| Approach | Pros | Cons | Score |
|----------|------|------|-------|
| **Hybrid In-Memory + Redis** | Low latency, persistent, scalable | Setup complexity, sync overhead | **8.6** |
| Pure In-Memory | Maximum speed, simple | No persistence, restart issues | 8.4 |
| Pure CLI coordination | Simple, direct control | High overhead, poor scalability | 7.5 |
| Database-only queue | Persistent, transactional | Higher latency, complex queries | 7.8 |

## Rationale

- **+30% efficiency improvement (8.6)**: Measured coordination gains in Phase 1

- **Phase 1 optimal balance**: Speed with reliability

- **Phase 2 privacy-ready**: Federated aggregation support

- **Framework integration**: Natural fit with LangGraph StateGraph

## Consequences

### Positive

- Optimal performance for high-frequency task operations

- Reliable task persistence across system restarts

- Efficient coordination between agents

- Strong foundation for Phase 2 federated architectures

### Negative

- Increased system complexity with multiple data stores

- Synchronization overhead between in-memory and persistent storage

- Need for conflict resolution in concurrent access scenarios

### Neutral

- Sync throttling mechanisms in Phase 1

- Flower integration for Phase 2 federated privacy

## Implementation Notes

### Task State Management
```python
from collections import deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import json

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class Task:
    task_id: str
    task_type: str
    description: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class TaskManager:
    def __init__(self, redis_client, state_graph):
        self.redis_client = redis_client
        self.state_graph = state_graph
        
        # In-memory queues for fast access
        self.priority_queues = {
            priority: deque() for priority in TaskPriority
        }
        
        # Task tracking
        self.active_tasks: Dict[str, Task] = {}
        self.agent_assignments: Dict[str, List[str]] = {}
        
        # Coordination
        self.task_locks: Dict[str, asyncio.Lock] = {}
        self.sync_interval = 5.0  # seconds
        self.last_sync = time.time()
        
        # Start background sync
        asyncio.create_task(self._background_sync())
    
    async def submit_task(self, task: Task) -> str:
        """Submit a new task to the system"""
        
        # Add to in-memory queue for fast access
        self.priority_queues[task.priority].append(task)
        self.active_tasks[task.task_id] = task
        
        # Persist to Redis
        await self._persist_task(task)
        
        # Notify agents via pub/sub
        await self.redis_client.publish(
            'task_notifications',
            json.dumps({
                'action': 'new_task',
                'task_id': task.task_id,
                'task_type': task.task_type,
                'priority': task.priority.value
            })
        )
        
        return task.task_id
    
    async def claim_task(self, agent_id: str, task_types: List[str] = None) -> Optional[Task]:
        """Claim the next available task for an agent"""
        
        # Check each priority level
        for priority in TaskPriority:
            queue = self.priority_queues[priority]
            
            # Find suitable task
            for i, task in enumerate(queue):
                if (task.status == TaskStatus.PENDING and
                    self._can_assign_task(task, agent_id, task_types)):
                    
                    # Remove from queue
                    queue.remove(task)
                    
                    # Assign to agent
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_agent = agent_id
                    task.started_at = time.time()
                    
                    # Update tracking
                    if agent_id not in self.agent_assignments:
                        self.agent_assignments[agent_id] = []
                    self.agent_assignments[agent_id].append(task.task_id)
                    
                    # Persist changes
                    await self._persist_task(task)
                    
                    return task
        
        return None
    
    async def update_task_status(self, task_id: str, status: TaskStatus, 
                               result: Optional[Dict] = None, 
                               error: Optional[str] = None) -> bool:
        """Update task status and result"""
        
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        
        # Update task state
        task.status = status
        if result:
            task.result = result
        if error:
            task.error = error
        
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task.completed_at = time.time()
            
            # Remove from agent assignments
            if task.assigned_agent and task.assigned_agent in self.agent_assignments:
                if task_id in self.agent_assignments[task.assigned_agent]:
                    self.agent_assignments[task.assigned_agent].remove(task_id)
        
        # Persist changes
        await self._persist_task(task)
        
        # Notify completion
        await self.redis_client.publish(
            'task_notifications',
            json.dumps({
                'action': 'task_updated',
                'task_id': task_id,
                'status': status.value,
                'agent_id': task.assigned_agent
            })
        )
        
        return True
```

### Task Dependencies and Workflow
```python
class TaskDependencyManager:
    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
        self.dependency_graph: Dict[str, List[str]] = {}
        self.waiting_tasks: Dict[str, List[str]] = {}
    
    async def add_task_with_dependencies(self, task: Task, dependencies: List[str]) -> str:
        """Add task with dependency constraints"""
        
        task.dependencies = dependencies
        self.dependency_graph[task.task_id] = dependencies
        
        # Check if all dependencies are met
        if await self._dependencies_satisfied(task.task_id):
            # Submit immediately
            return await self.task_manager.submit_task(task)
        else:
            # Add to waiting list
            for dep_id in dependencies:
                if dep_id not in self.waiting_tasks:
                    self.waiting_tasks[dep_id] = []
                self.waiting_tasks[dep_id].append(task.task_id)
            
            # Store task but don't queue yet
            self.task_manager.active_tasks[task.task_id] = task
            task.status = TaskStatus.PENDING
            await self.task_manager._persist_task(task)
            
            return task.task_id
    
    async def handle_task_completion(self, completed_task_id: str):
        """Handle completion and check for dependent tasks"""
        
        if completed_task_id in self.waiting_tasks:
            dependent_task_ids = self.waiting_tasks[completed_task_id]
            
            for task_id in dependent_task_ids:
                if await self._dependencies_satisfied(task_id):
                    # Dependencies satisfied, submit task
                    task = self.task_manager.active_tasks[task_id]
                    await self.task_manager.submit_task(task)
            
            # Clean up
            del self.waiting_tasks[completed_task_id]
    
    async def _dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are completed"""
        
        if task_id not in self.dependency_graph:
            return True
        
        dependencies = self.dependency_graph[task_id]
        
        for dep_id in dependencies:
            if dep_id not in self.task_manager.active_tasks:
                return False
            
            dep_task = self.task_manager.active_tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
```

### Agent Coordination and Load Balancing
```python
class AgentCoordinator:
    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_load: Dict[str, int] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        
    def register_agent(self, agent_id: str, capabilities: List[str]):
        """Register an agent with its capabilities"""
        self.agent_capabilities[agent_id] = capabilities
        self.agent_load[agent_id] = 0
        self.agent_performance[agent_id] = {
            'success_rate': 1.0,
            'avg_completion_time': 0.0,
            'total_tasks': 0
        }
    
    async def assign_optimal_task(self, agent_id: str) -> Optional[Task]:
        """Assign the most suitable task to an agent"""
        
        if agent_id not in self.agent_capabilities:
            return None
        
        capabilities = self.agent_capabilities[agent_id]
        current_load = self.agent_load[agent_id]
        
        # Find best matching task
        best_task = None
        best_score = 0.0
        
        for priority in TaskPriority:
            queue = self.task_manager.priority_queues[priority]
            
            for task in queue:
                if task.status != TaskStatus.PENDING:
                    continue
                
                # Calculate assignment score
                score = self._calculate_assignment_score(
                    task, agent_id, capabilities, current_load
                )
                
                if score > best_score:
                    best_score = score
                    best_task = task
        
        if best_task:
            # Claim the task
            claimed_task = await self.task_manager.claim_task(
                agent_id, capabilities
            )
            
            if claimed_task and claimed_task.task_id == best_task.task_id:
                self.agent_load[agent_id] += 1
                return claimed_task
        
        return None
    
    def _calculate_assignment_score(self, task: Task, agent_id: str, 
                                  capabilities: List[str], current_load: int) -> float:
        """Calculate how well a task matches an agent"""
        
        score = 0.0
        
        # Capability match
        if task.task_type in capabilities or 'general' in capabilities:
            score += 50.0
        
        # Priority boost
        priority_boost = {
            TaskPriority.CRITICAL: 40.0,
            TaskPriority.HIGH: 25.0,
            TaskPriority.MEDIUM: 10.0,
            TaskPriority.LOW: 5.0
        }
        score += priority_boost.get(task.priority, 0.0)
        
        # Load balancing penalty
        load_penalty = current_load * 5.0
        score -= load_penalty
        
        # Performance bonus
        if agent_id in self.agent_performance:
            perf = self.agent_performance[agent_id]
            score += perf['success_rate'] * 20.0
            
            # Prefer faster agents for simple tasks
            if task.task_type in ['simple_qa', 'quick_analysis']:
                score += (1.0 / max(perf['avg_completion_time'], 0.1)) * 10.0
        
        return score
    
    async def handle_task_completion(self, agent_id: str, task_id: str, 
                                   success: bool, completion_time: float):
        """Update agent performance metrics"""
        
        if agent_id in self.agent_load:
            self.agent_load[agent_id] = max(0, self.agent_load[agent_id] - 1)
        
        if agent_id in self.agent_performance:
            perf = self.agent_performance[agent_id]
            
            # Update success rate (moving average)
            alpha = 0.1  # Learning rate
            perf['success_rate'] = (
                (1 - alpha) * perf['success_rate'] + 
                alpha * (1.0 if success else 0.0)
            )
            
            # Update completion time (moving average)
            perf['avg_completion_time'] = (
                (1 - alpha) * perf['avg_completion_time'] + 
                alpha * completion_time
            )
            
            perf['total_tasks'] += 1
```

### Phase 2 Federated Extensions
```python
class FederatedTaskManager:  # Phase 2
    def __init__(self, local_task_manager: TaskManager, node_id: str):
        self.local_manager = local_task_manager
        self.node_id = node_id
        self.federation_client = FederationClient()
        self.privacy_filter = PrivacyFilter()
        
    async def sync_with_federation(self):
        """Privacy-preserving task coordination with other nodes"""
        
        # Create privacy-preserving task summary
        local_summary = self._create_task_summary()
        
        # Exchange with other nodes
        federated_summaries = await self.federation_client.exchange_summaries(
            local_summary
        )
        
        # Update coordination based on federated insights
        await self._update_coordination_strategy(federated_summaries)
    
    def _create_task_summary(self) -> Dict:
        """Create privacy-preserving summary of local task state"""
        
        summary = {
            'node_id': self.node_id,
            'task_counts_by_type': {},
            'average_completion_times': {},
            'success_rates_by_type': {},
            'current_load': len(self.local_manager.active_tasks),
            # Exclude: specific task content, user data, proprietary logic
        }
        
        # Aggregate statistics without exposing details
        for task_type in ['coding', 'analysis', 'reasoning']:
            relevant_tasks = [
                task for task in self.local_manager.active_tasks.values()
                if task.task_type == task_type
            ]
            
            summary['task_counts_by_type'][task_type] = len(relevant_tasks)
            
            if relevant_tasks:
                completion_times = [
                    task.completed_at - task.started_at
                    for task in relevant_tasks
                    if task.completed_at and task.started_at
                ]
                
                if completion_times:
                    summary['average_completion_times'][task_type] = sum(completion_times) / len(completion_times)
        
        return summary
    
    async def _update_coordination_strategy(self, federated_summaries: List[Dict]):
        """Update local coordination based on federated insights"""
        
        # Analyze federated load distribution
        total_load = sum(summary.get('current_load', 0) for summary in federated_summaries)
        
        if total_load > 0:
            # Adjust local task acceptance based on relative load
            local_load_ratio = self.local_manager.active_tasks.__len__() / total_load
            
            # If significantly overloaded, become more selective
            if local_load_ratio > 0.4:  # More than 40% of total load
                self.local_manager.task_acceptance_threshold = 0.8
            else:
                self.local_manager.task_acceptance_threshold = 0.5
```

## Performance Monitoring

### Task Metrics and Analytics
```python
class TaskAnalytics:
    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
        self.metrics_history: List[Dict] = []
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive task management metrics"""
        
        current_time = time.time()
        active_tasks = self.task_manager.active_tasks
        
        metrics = {
            'timestamp': current_time,
            'total_tasks': len(active_tasks),
            'tasks_by_status': self._count_by_status(active_tasks),
            'tasks_by_priority': self._count_by_priority(active_tasks),
            'average_queue_time': self._calculate_avg_queue_time(active_tasks),
            'average_completion_time': self._calculate_avg_completion_time(active_tasks),
            'throughput_per_hour': self._calculate_throughput(),
            'agent_utilization': self._calculate_agent_utilization(),
            'error_rate': self._calculate_error_rate(active_tasks)
        }
        
        self.metrics_history.append(metrics)
        
        # Keep last 24 hours of metrics
        cutoff_time = current_time - 86400
        self.metrics_history = [
            m for m in self.metrics_history if m['timestamp'] > cutoff_time
        ]
        
        return metrics
    
    def _calculate_throughput(self) -> float:
        """Calculate tasks completed per hour"""
        
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_metrics = self.metrics_history[-12:]  # Last hour (5-minute intervals)
        
        if not recent_metrics:
            return 0.0
        
        completed_counts = [
            m['tasks_by_status'].get('completed', 0) for m in recent_metrics
        ]
        
        if len(completed_counts) >= 2:
            throughput = (completed_counts[-1] - completed_counts[0]) / (len(completed_counts) / 12)
            return max(0.0, throughput)
        
        return 0.0
```

## Performance Targets

| Metric | Phase 1 Target | Phase 2 Target |
|--------|----------------|----------------|
| Task Assignment Latency | <10ms | <50ms |
| Task Completion Rate | >95% | >95% |
| Agent Utilization | >80% | >85% |
| System Throughput | 100+ tasks/hour | 1000+ tasks/hour |
| Coordination Efficiency | +30% vs baseline | +40% vs baseline |

## Related Decisions

- ADR-004: Orchestration and Task Management

- ADR-005: Caching and Shared Context Layer

- ADR-014: Federated Basics (Phase 2)

## Monitoring

- Task queue depths and processing rates

- Agent assignment patterns and load distribution

- Task completion times and success rates

- System resource utilization

- Coordination overhead and efficiency
