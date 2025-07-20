# Quick Start Guide

Get up and running with Temporal Bandwidth Optimizer in 5 minutes.

## Installation

```bash
pip install temporal-bandwidth-optimizer
```

Or for development:

```bash
git clone https://github.com/temporal-bandwidth-optimizer/temporal-bandwidth-optimizer
cd temporal-bandwidth-optimizer
pip install -e .[dev,research]
```

## Basic Usage

### 1. Simple Optimization (2 minutes)

```python
import asyncio
from tboptimizer import ClaudeOptimizedClient, CollaborationContext

async def quick_demo():
    # Initialize optimized client
    client = ClaudeOptimizedClient(
        api_key="your_anthropic_api_key",
        optimization_level="balanced"
    )
    
    # Create collaboration context
    context = CollaborationContext(
        session_id="demo_session",
        task_type="analysis"
    )
    
    # Collaborate with optimization
    response = await client.collaborate(
        messages=[{"role": "user", "content": "Help me analyze market trends"}],
        context=context
    )
    
    print(f"Response: {response.content}")
    print(f"Optimization applied: {response.optimization_applied}")
    print(f"Response time: {response.response_time:.2f}s")
    
    return response

# Run the demo
result = asyncio.run(quick_demo())
```

### 2. Measure Bandwidth Efficiency (3 minutes)

```python
from tboptimizer import BandwidthMonitor

# Initialize monitoring
monitor = BandwidthMonitor()

# Track interactions
for turn in range(5):
    metrics = monitor.track_interaction(
        response_time=1.2,  # Simulated response time
        turn_count=turn + 1,
        context_length=100,
        task_complexity=3
    )
    
    print(f"Turn {turn + 1}: Efficiency = {metrics['efficiency']:.3f}")

# Generate report
report = monitor.generate_performance_report()
print(f"\nFinal efficiency: {report.turns_per_second:.3f} turns/second")
print(f"Bandwidth degradation: {report.bandwidth_degradation:.1%}")
```

### 3. Run Research Demo (5 minutes)

```bash
python examples/research_collaboration_demo.py
```

Expected output:
```
🔬 Temporal Bandwidth Optimization Research Demo
==================================================

📊 Testing baseline condition...
  - strategy_planning: 4.2s (efficiency: 0.095)
  - technical_analysis: 3.8s (efficiency: 0.105)

📊 Testing optimized condition...  
  - strategy_planning: 2.8s (efficiency: 0.143)
  - technical_analysis: 2.4s (efficiency: 0.167)

✅ SIGNIFICANT improvement detected!
   Efficiency improvement: +42.3%
   Time reduction: 31.2% faster
```

## Configuration Options

### Optimization Levels

```python
from tboptimizer import OptimizationLevel

# Speed: Maximum performance, minimum latency
client = ClaudeOptimizedClient(optimization_level="speed")

# Balanced: Balance speed and quality (recommended)
client = ClaudeOptimizedClient(optimization_level="balanced")

# Quality: Favor quality over speed
client = ClaudeOptimizedClient(optimization_level="quality")

# Research: Optimized for research workflows
client = ClaudeOptimizedClient(optimization_level="research")
```

### Advanced Configuration

```python
client = ClaudeOptimizedClient(
    optimization_level="balanced",
    target_latency=0.5,           # Target response time
    cache_size=5000,              # Cache size
    enable_monitoring=True,       # Real-time monitoring
)
```

## Next Steps

### For Researchers
- Run full validation study: `python research/validation_study.py`
- Explore benchmark datasets: `python research/benchmark_datasets.py`
- See [Research Methods](research_methods.md)

### For Production
- Try multi-agent coordination: `python examples/agent_coordination_benchmark.py`
- Production deployment: `python examples/production_deployment_example.py`
- See [Deployment Guide](deployment_guide.md)

### For Development
- Run tests: `pytest tests/`
- See [API Reference](api_reference.md)
- Check [Performance Tuning](performance_tuning.md)

## Troubleshooting

### Common Issues

**1. API Key Not Found**
```bash
export ANTHROPIC_API_KEY="your_key_here"
```

**2. Import Errors**
```bash
pip install anthropic numpy scipy pandas
```

**3. Async Errors**
Make sure to use `await` with async functions:
```python
# Correct
response = await client.collaborate(messages, context)

# Incorrect  
response = client.collaborate(messages, context)
```

### Performance Issues

If you see poor performance:

1. Check network latency to API endpoints
2. Verify optimization level is appropriate
3. Monitor cache hit rates
4. Review target latency settings

### Getting Help

- [GitHub Issues](https://github.com/temporal-bandwidth-optimizer/temporal-bandwidth-optimizer/issues)
- [API Reference](api_reference.md)
- [FAQ](faq.md)

---

**You're ready!** The system is now optimizing your AI collaboration bandwidth. Monitor the efficiency metrics to see the improvements.