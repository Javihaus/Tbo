# API Reference

Complete reference for Temporal Bandwidth Optimizer API.

## Core Classes

### ClaudeOptimizedClient

Drop-in replacement for Claude API with bandwidth optimization.

```python
class ClaudeOptimizedClient:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 optimization_level: Union[str, OptimizationLevel] = "balanced",
                 enable_monitoring: bool = True,
                 cache_size: int = 1000,
                 target_latency: float = 0.5)
```

**Parameters:**
- `api_key`: Anthropic API key (uses environment variable if None)
- `optimization_level`: Optimization preset ("speed", "balanced", "quality", "research")
- `enable_monitoring`: Enable real-time bandwidth monitoring
- `cache_size`: Maximum cache entries
- `target_latency`: Target response latency in seconds

#### Methods

##### `collaborate(messages, context, model="claude-3-sonnet-20240229", **kwargs) -> OptimizedResponse`

Main collaborative interface with optimization.

```python
async def collaborate(self, 
                     messages: List[Dict[str, str]],
                     context: CollaborationContext,
                     model: str = "claude-3-sonnet-20240229",
                     max_tokens: int = 1000,
                     **kwargs) -> OptimizedResponse
```

**Parameters:**
- `messages`: List of conversation messages
- `context`: Collaboration context and metadata
- `model`: Claude model to use
- `max_tokens`: Maximum response tokens
- `**kwargs`: Additional Claude API parameters

**Returns:** `OptimizedResponse` with optimization metrics

**Example:**
```python
response = await client.collaborate(
    messages=[{"role": "user", "content": "Analyze this data"}],
    context=CollaborationContext(session_id="123", task_type="analysis"),
    max_tokens=500
)
```

##### `get_performance_report() -> Dict[str, Any]`

Get comprehensive performance report.

```python
def get_performance_report(self) -> Dict[str, Any]
```

**Returns:** Dictionary with performance metrics including:
- Request statistics
- Engine performance
- Bandwidth metrics
- Temporal analysis

##### `export_research_data(format="dict") -> Any`

Export data for research analysis.

```python
def export_research_data(self, format: str = "dict") -> Any
```

**Parameters:**
- `format`: Export format ("dict", "pandas", "numpy")

**Returns:** Data in requested format

### BandwidthMonitor

Real-time measurement of collaborative reasoning efficiency.

```python
class BandwidthMonitor:
    def __init__(self, 
                 window_size: int = 50,
                 degradation_threshold: float = 2.0,
                 baseline_efficiency: Optional[float] = None)
```

**Parameters:**
- `window_size`: Rolling window size for efficiency calculations
- `degradation_threshold`: Circuit breaker threshold in seconds
- `baseline_efficiency`: Reference efficiency for comparison

#### Methods

##### `track_interaction(response_time, turn_count, **kwargs) -> Dict[str, Any]`

Track a single collaborative interaction.

```python
def track_interaction(self, 
                     response_time: float, 
                     turn_count: int,
                     context_length: int = 0,
                     task_complexity: int = 1) -> Dict[str, Any]
```

**Parameters:**
- `response_time`: Time for AI response in seconds
- `turn_count`: Current turn number in conversation
- `context_length`: Length of conversation context
- `task_complexity`: Task complexity rating (1-5)

**Returns:** Dictionary with immediate metrics and degradation status

##### `calculate_efficiency_ratio() -> float`

Calculate current collaborative efficiency (turns per second).

##### `detect_degradation_threshold(response_time) -> bool`

Detect if response time exceeds degradation threshold.

##### `generate_performance_report() -> CollaborationMetrics`

Generate comprehensive performance report.

### OptimizationEngine

Applies empirically-validated optimization strategies.

```python
class OptimizationEngine:
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
                 target_latency: float = 0.5,
                 cache_size: int = 1000)
```

#### Methods

##### `optimize_request_routing(context, available_endpoints) -> RoutingDecision`

Optimize request routing for minimum latency.

```python
def optimize_request_routing(self, 
                            context: Dict[str, Any],
                            available_endpoints: List[str]) -> RoutingDecision
```

##### `apply_predictive_caching(conversation_state) -> CacheStrategy`

Apply predictive caching based on conversation patterns.

```python
def apply_predictive_caching(self, 
                            conversation_state: Dict[str, Any]) -> CacheStrategy
```

##### `coordinate_multi_agent_timing(agents, coordination_strategy) -> TimingPlan`

Coordinate timing for multi-agent collaborative tasks.

```python
def coordinate_multi_agent_timing(self, 
                                agents: List[Dict[str, Any]],
                                coordination_strategy: str = "parallel") -> TimingPlan
```

### TemporalAnalyzer

Advanced temporal analysis for collaborative reasoning systems.

```python
class TemporalAnalyzer:
    def __init__(self, sampling_rate: float = 1.0, analysis_window: int = 100)
```

#### Methods

##### `add_measurement(timestamp, response_time, efficiency, turn_interval)`

Add new temporal measurement for analysis.

##### `detect_collaborative_rhythm() -> RhythmAnalysis`

Detect natural collaborative rhythm patterns.

##### `detect_degradation_patterns() -> List[TemporalPattern]`

Detect temporal degradation patterns in efficiency data.

##### `analyze_latency_impact(latency_values) -> Dict[str, Any]`

Analyze impact of latency on collaborative patterns.

## Data Classes

### CollaborationContext

Context for collaborative AI interactions.

```python
@dataclass
class CollaborationContext:
    session_id: str
    user_id: Optional[str] = None
    task_type: str = "general"
    complexity: int = 1
    priority: int = 5
    conversation_history: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
```

### OptimizedResponse

Enhanced response with optimization metrics.

```python
@dataclass
class OptimizedResponse:
    content: str
    response_time: float
    cache_hit: bool
    optimization_applied: bool
    bandwidth_metrics: Dict[str, Any]
    routing_decision: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    
    # Claude API compatibility
    @property
    def text(self) -> str
```

### CollaborationMetrics

Comprehensive metrics for collaborative reasoning performance.

```python
@dataclass
class CollaborationMetrics:
    turns_per_second: float
    total_turns: int
    total_time: float
    efficiency_ratio: float
    response_times: List[float]
    turn_gaps: List[float]
    bandwidth_degradation: float
    statistical_significance: float
    confidence_interval: tuple
```

### RoutingDecision

Represents a routing decision for request optimization.

```python
@dataclass
class RoutingDecision:
    target_endpoint: str
    predicted_latency: float
    confidence: float
    strategy_used: RoutingStrategy
    cache_recommendation: str
    fallback_options: List[str] = field(default_factory=list)
```

### CacheStrategy

Cache strategy for response optimization.

```python
@dataclass
class CacheStrategy:
    cache_key: str
    cache_hit: bool
    generation_strategy: str  # "exact", "semantic", "predictive"
    confidence: float
    ttl: int  # Time to live in seconds
    preload_recommendations: List[str] = field(default_factory=list)
```

## Enums

### OptimizationLevel

Optimization presets for different use cases.

```python
class OptimizationLevel(Enum):
    SPEED = "speed"          # Maximum speed, minimum latency
    BALANCED = "balanced"    # Balance speed and quality  
    QUALITY = "quality"      # Favor quality over speed
    RESEARCH = "research"    # Optimized for research workflows
```

### RoutingStrategy

Request routing strategies.

```python
class RoutingStrategy(Enum):
    LATENCY_PRIORITY = "latency_priority"
    LOAD_BALANCE = "load_balance"
    CAPABILITY_MATCH = "capability_match"
    ADAPTIVE = "adaptive"
```

## Integrations

### OpenAI Integration

```python
from tboptimizer import OpenAIOptimizedClient

client = OpenAIOptimizedClient(
    api_key="your_openai_key",
    optimization_level="balanced"
)

response = await client.collaborate(
    messages=[{"role": "user", "content": "Help me code"}],
    context=CollaborationContext(session_id="openai_session"),
    model="gpt-3.5-turbo"
)
```

### Generic LLM Integration

```python
from tboptimizer import GenericLLMClient, LLMProvider

class CustomProvider(LLMProvider):
    async def make_request(self, messages, **kwargs):
        # Custom implementation
        return "Custom response"
    
    def get_provider_name(self):
        return "custom_api"

provider = CustomProvider()
client = GenericLLMClient(provider, optimization_level="research")
```

## Research Tools

### Validation Study

```python
from research.validation_study import ValidationStudy

study = ValidationStudy(api_key="your_key")
results = await study.run_full_study(tasks_per_condition=50)
```

### Benchmark Datasets

```python
from research.benchmark_datasets import BenchmarkDatasets

dataset = BenchmarkDatasets()
tasks = dataset.get_balanced_sample(tasks_per_domain=5)
dataset.export_dataset("tasks.json")
```

## Error Handling

### Common Exceptions

- `TimeoutError`: Request exceeded timeout
- `ValueError`: Invalid parameters
- `ImportError`: Missing dependencies
- `Exception`: General API errors

### Error Response Format

```python
{
    "success": False,
    "error": "Error description",
    "response_time": 2.5,
    "request_id": "req_123"
}
```

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY`: Anthropic API key
- `OPENAI_API_KEY`: OpenAI API key
- `TBO_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `TBO_CACHE_SIZE`: Default cache size
- `TBO_TARGET_LATENCY`: Default target latency

### Configuration File

```yaml
# tboptimizer.yaml
optimization:
  level: "balanced"
  target_latency: 0.5
  
caching:
  size: 5000
  ttl: 1800
  
monitoring:
  enabled: true
  window_size: 100
```

## Performance Monitoring

### Metrics Export

```python
# Prometheus format
metrics = client.get_performance_report()
for metric, value in metrics.items():
    print(f"tboptimizer_{metric} {value}")

# Custom monitoring
def alert_on_degradation(client):
    report = client.get_performance_report()
    if report["bandwidth_metrics"]["bandwidth_degradation"] > 0.3:
        send_alert("High bandwidth degradation detected")
```

### Health Checks

```python
# Health check endpoint
async def health_check():
    try:
        test_response = await client.collaborate(
            messages=[{"role": "user", "content": "health check"}],
            context=CollaborationContext(session_id="health"),
            max_tokens=10
        )
        return {"status": "healthy", "response_time": test_response.response_time}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

---

For more examples and advanced usage, see the [examples/](../examples/) directory.