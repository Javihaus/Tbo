"""Pytest configuration and shared fixtures for TBO tests."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

# Import TBO components for fixtures
from src.tboptimizer.core.bandwidth_monitor import BandwidthMonitor
from src.tboptimizer.core.optimization_engine import OptimizationEngine, OptimizationLevel
from src.tboptimizer.integrations.claude_adapter import CollaborationContext


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def bandwidth_monitor():
    """Fixture providing configured BandwidthMonitor."""
    return BandwidthMonitor(
        window_size=10,
        degradation_threshold=2.0,
        baseline_efficiency=0.125
    )


@pytest.fixture
def optimization_engine():
    """Fixture providing configured OptimizationEngine."""
    return OptimizationEngine(
        optimization_level=OptimizationLevel.BALANCED,
        target_latency=0.5,
        cache_size=100
    )


@pytest.fixture
def sample_collaboration_context():
    """Fixture providing sample collaboration context."""
    return CollaborationContext(
        session_id="test_session_123",
        user_id="test_user",
        task_type="analysis",
        complexity=3,
        priority=2,
        metadata={"test": True}
    )


@pytest.fixture
def sample_messages():
    """Fixture providing sample conversation messages."""
    return [
        {"role": "user", "content": "Help me analyze this data"},
        {"role": "assistant", "content": "I'll help you analyze the data. What specific aspects are you interested in?"},
        {"role": "user", "content": "Focus on trends and patterns"}
    ]


@pytest.fixture
def mock_anthropic_client():
    """Fixture providing mocked Anthropic client."""
    with patch('src.tboptimizer.integrations.claude_adapter.anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_anthropic.Client.return_value = mock_client
        
        # Mock response structure
        mock_response = Mock()
        mock_response.content = "Mocked response content"
        mock_client.messages.create.return_value = mock_response
        
        yield mock_client


@pytest.fixture
def mock_openai_client():
    """Fixture providing mocked OpenAI client."""
    with patch('src.tboptimizer.integrations.openai_adapter.openai') as mock_openai:
        mock_client = Mock()
        mock_openai.AsyncOpenAI.return_value = mock_client
        
        # Mock response structure
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Mocked OpenAI response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        async def mock_create(*args, **kwargs):
            return mock_response
        
        mock_client.chat.completions.create = mock_create
        
        yield mock_client


@pytest.fixture
def performance_test_data():
    """Fixture providing performance test data."""
    return {
        "response_times": [0.3, 0.5, 0.4, 0.6, 0.7, 0.2, 0.8, 0.4, 0.5, 0.3],
        "turn_counts": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "context_lengths": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        "task_complexities": [1, 2, 3, 2, 4, 1, 5, 3, 2, 4]
    }


@pytest.fixture
def research_validation_data():
    """Fixture providing research validation data."""
    return {
        "baseline_condition": {
            "artificial_delay": 0.0,
            "expected_degradation": 0.0,
            "response_times": [0.8, 0.9, 0.7, 0.8, 0.9]
        },
        "low_delay_condition": {
            "artificial_delay": 2.0,
            "expected_degradation": 0.21,
            "response_times": [2.8, 2.9, 2.7, 2.8, 2.9]
        },
        "medium_delay_condition": {
            "artificial_delay": 5.0,
            "expected_degradation": 0.41,
            "response_times": [5.8, 5.9, 5.7, 5.8, 5.9]
        },
        "high_delay_condition": {
            "artificial_delay": 10.0,
            "expected_degradation": 0.56,
            "response_times": [10.8, 10.9, 10.7, 10.8, 10.9]
        }
    }


@pytest.fixture
def cache_test_data():
    """Fixture providing cache test data."""
    return {
        "conversation_states": [
            {
                "messages": [{"role": "user", "content": "analyze market trends"}],
                "context": "business_analysis",
                "task_type": "analysis"
            },
            {
                "messages": [{"role": "user", "content": "analyze market patterns"}],
                "context": "business_analysis", 
                "task_type": "analysis"
            },
            {
                "messages": [{"role": "user", "content": "review technical documentation"}],
                "context": "technical_review",
                "task_type": "review"
            }
        ],
        "expected_cache_keys": [
            "market_analysis_key",
            "market_patterns_key", 
            "tech_review_key"
        ],
        "cached_responses": [
            "Market analysis results...",
            "Market pattern insights...",
            "Technical review complete..."
        ]
    }


@pytest.fixture
def routing_test_scenarios():
    """Fixture providing routing test scenarios."""
    return {
        "high_priority": {
            "context": {"priority": 1, "complexity": 3, "type": "urgent"},
            "endpoints": ["primary_fast", "primary_standard", "backup"],
            "expected_strategy": "latency_priority"
        },
        "load_balanced": {
            "context": {"priority": 5, "complexity": 2, "type": "standard"},
            "endpoints": ["overloaded_ep", "normal_ep", "light_ep"],
            "expected_strategy": "load_balance"
        },
        "adaptive": {
            "context": {"priority": 3, "complexity": 4, "type": "complex"},
            "endpoints": ["specialist_ep", "general_ep", "backup_ep"],
            "expected_strategy": "adaptive"
        }
    }


@pytest.fixture
def timing_coordination_scenarios():
    """Fixture providing multi-agent timing scenarios."""
    return {
        "parallel_agents": [
            {"id": "agent_a", "estimated_duration": 2.0, "dependencies": []},
            {"id": "agent_b", "estimated_duration": 1.5, "dependencies": []},
            {"id": "agent_c", "estimated_duration": 2.5, "dependencies": []}
        ],
        "sequential_agents": [
            {"id": "step_1", "estimated_duration": 1.0, "dependencies": []},
            {"id": "step_2", "estimated_duration": 1.5, "dependencies": ["step_1"]},
            {"id": "step_3", "estimated_duration": 2.0, "dependencies": ["step_2"]}
        ],
        "complex_dependencies": [
            {"id": "foundation", "estimated_duration": 2.0, "dependencies": []},
            {"id": "analysis", "estimated_duration": 1.5, "dependencies": ["foundation"]},
            {"id": "synthesis", "estimated_duration": 1.0, "dependencies": ["foundation"]},
            {"id": "report", "estimated_duration": 1.5, "dependencies": ["analysis", "synthesis"]}
        ]
    }


@pytest.fixture
def metrics_assertions():
    """Fixture providing metric assertion helpers."""
    class MetricsAssertions:
        @staticmethod
        def assert_efficiency_metrics(metrics, min_efficiency=0.0, max_efficiency=1.0):
            """Assert efficiency metrics are within valid ranges."""
            assert min_efficiency <= metrics.get("efficiency", 0) <= max_efficiency
            assert metrics.get("response_time", 0) > 0
        
        @staticmethod
        def assert_bandwidth_metrics(metrics, expected_degradation_range=(0.0, 1.0)):
            """Assert bandwidth metrics are valid."""
            degradation = metrics.get("bandwidth_degradation", 0)
            assert expected_degradation_range[0] <= degradation <= expected_degradation_range[1]
        
        @staticmethod
        def assert_cache_metrics(metrics, expected_hit_rate_range=(0.0, 1.0)):
            """Assert cache metrics are valid."""
            hit_rate = metrics.get("cache_hit_rate", 0)
            assert expected_hit_rate_range[0] <= hit_rate <= expected_hit_rate_range[1]
        
        @staticmethod
        def assert_timing_metrics(timing_plan, expected_duration_range=(0.0, float('inf'))):
            """Assert timing plan metrics are valid."""
            assert expected_duration_range[0] <= timing_plan.total_duration <= expected_duration_range[1]
            assert len(timing_plan.agent_schedules) > 0
    
    return MetricsAssertions()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "research: marks tests as research validation tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark research tests
        if "research" in item.nodeid or "validation" in item.nodeid:
            item.add_marker(pytest.mark.research)
        
        # Mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)


# Test data helpers
class TestDataGenerator:
    """Helper class for generating test data."""
    
    @staticmethod
    def generate_response_times(count=10, base_time=0.5, variance=0.2):
        """Generate realistic response times."""
        import random
        return [base_time + random.uniform(-variance, variance) for _ in range(count)]
    
    @staticmethod
    def generate_conversation_history(length=5):
        """Generate conversation history."""
        history = []
        for i in range(length):
            if i % 2 == 0:
                history.append({"role": "user", "content": f"User message {i//2 + 1}"})
            else:
                history.append({"role": "assistant", "content": f"Assistant response {i//2 + 1}"})
        return history
    
    @staticmethod
    def generate_efficiency_data(baseline=0.125, degradation_factor=0.0):
        """Generate efficiency data with optional degradation."""
        return baseline * (1.0 - degradation_factor)


@pytest.fixture
def test_data_generator():
    """Fixture providing test data generator."""
    return TestDataGenerator()