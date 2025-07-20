"""Tests for ClaudeOptimizedClient with API integration validation."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.tboptimizer.integrations.claude_adapter import (
    ClaudeOptimizedClient,
    CollaborationContext,
    OptimizedResponse,
    OptimizationLevel
)


class TestCollaborationContext:
    """Test suite for CollaborationContext data class."""
    
    def test_context_creation(self):
        """Test context creation with defaults."""
        context = CollaborationContext(session_id="test_session")
        
        assert context.session_id == "test_session"
        assert context.user_id is None
        assert context.task_type == "general"
        assert context.complexity == 1
        assert context.priority == 5
        assert context.conversation_history == []
        assert context.metadata == {}
    
    def test_context_with_full_data(self):
        """Test context creation with all fields."""
        context = CollaborationContext(
            session_id="session_123",
            user_id="user_456",
            task_type="analysis",
            complexity=4,
            priority=2,
            conversation_history=[{"role": "user", "content": "Hello"}],
            metadata={"experiment": True}
        )
        
        assert context.session_id == "session_123"
        assert context.user_id == "user_456"
        assert context.task_type == "analysis"
        assert context.complexity == 4
        assert context.priority == 2
        assert len(context.conversation_history) == 1
        assert context.metadata["experiment"]


class TestOptimizedResponse:
    """Test suite for OptimizedResponse data class."""
    
    def test_response_creation(self):
        """Test response creation."""
        response = OptimizedResponse(
            content="Test response content",
            response_time=0.45,
            cache_hit=True,
            optimization_applied=True,
            bandwidth_metrics={"efficiency": 0.15, "degradation": 0.1},
            routing_decision={"endpoint": "primary"},
            confidence=0.95
        )
        
        assert response.content == "Test response content"
        assert response.response_time == 0.45
        assert response.cache_hit
        assert response.optimization_applied
        assert response.confidence == 0.95
    
    def test_claude_api_compatibility(self):
        """Test Claude API compatibility properties."""
        response = OptimizedResponse(
            content="Compatible response",
            response_time=0.3,
            cache_hit=False,
            optimization_applied=True,
            bandwidth_metrics={}
        )
        
        # Test .text property for Claude API compatibility
        assert response.text == "Compatible response"
    
    def test_response_to_dict(self):
        """Test response dictionary conversion."""
        response = OptimizedResponse(
            content="Dict test",
            response_time=0.2,
            cache_hit=False,
            optimization_applied=True,
            bandwidth_metrics={"turns_per_second": 0.12}
        )
        
        response_dict = response.to_dict()
        
        assert isinstance(response_dict, dict)
        assert response_dict["content"] == "Dict test"
        assert response_dict["response_time"] == 0.2
        assert not response_dict["cache_hit"]


class TestClaudeOptimizedClient:
    """Test suite for ClaudeOptimizedClient functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock anthropic module to avoid import requirements
        self.mock_anthropic = Mock()
        self.mock_claude_client = Mock()
        self.mock_anthropic.Client.return_value = self.mock_claude_client
        
        # Patch anthropic import
        self.anthropic_patcher = patch('src.tboptimizer.integrations.claude_adapter.anthropic', self.mock_anthropic)
        self.anthropic_patcher.start()
        
        # Create client
        self.client = ClaudeOptimizedClient(
            api_key="test_key",
            optimization_level="balanced",
            enable_monitoring=True,
            cache_size=50,
            target_latency=0.5
        )
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.anthropic_patcher.stop()
    
    def test_client_initialization(self):
        """Test client initialization."""
        assert self.client.optimization_engine.optimization_level == OptimizationLevel.BALANCED
        assert self.client.optimization_engine.target_latency == 0.5
        assert self.client.bandwidth_monitor is not None
        assert self.client.temporal_analyzer is not None
        assert len(self.client.turn_counts) == 0
        assert self.client.total_requests == 0
    
    def test_optimization_level_string_conversion(self):
        """Test optimization level string to enum conversion."""
        # Test with string input
        client = ClaudeOptimizedClient(
            api_key="test_key",
            optimization_level="research"
        )
        
        assert client.optimization_engine.optimization_level == OptimizationLevel.RESEARCH
    
    @pytest.mark.asyncio
    async def test_collaborate_cache_hit(self):
        """Test collaboration with cache hit."""
        # Setup cache hit scenario
        messages = [{"role": "user", "content": "Test message"}]
        context = CollaborationContext(session_id="test_session")
        
        # Mock cache hit
        with patch.object(self.client.optimization_engine, 'apply_predictive_caching') as mock_cache:
            mock_cache.return_value = Mock(
                cache_hit=True,
                cache_key="test_key",
                confidence=1.0
            )
            
            # Mock cached response
            self.client.optimization_engine.response_cache["test_key"] = {
                "response": "Cached response"
            }
            
            response = await self.client.collaborate(messages, context)
            
            assert isinstance(response, OptimizedResponse)
            assert response.content == "Cached response"
            assert response.cache_hit
            assert response.optimization_applied
    
    @pytest.mark.asyncio
    async def test_collaborate_api_call(self):
        """Test collaboration with actual API call."""
        messages = [{"role": "user", "content": "Test API call"}]
        context = CollaborationContext(session_id="api_test_session")
        
        # Mock API response
        mock_api_response = Mock()
        mock_api_response.content = "API response content"
        
        # Mock cache miss
        with patch.object(self.client.optimization_engine, 'apply_predictive_caching') as mock_cache:
            mock_cache.return_value = Mock(
                cache_hit=False,
                cache_key="api_key",
                confidence=0.0
            )
            
            # Mock routing decision
            with patch.object(self.client.optimization_engine, 'optimize_request_routing') as mock_routing:
                mock_routing.return_value = Mock(
                    target_endpoint="claude-primary",
                    predicted_latency=0.3,
                    to_dict=lambda: {"endpoint": "claude-primary"}
                )
                
                # Mock Claude API call
                with patch.object(self.client, '_make_claude_request', new_callable=AsyncMock) as mock_api:
                    mock_api.return_value = mock_api_response
                    
                    response = await self.client.collaborate(messages, context, max_tokens=100)
                    
                    assert isinstance(response, OptimizedResponse)
                    assert response.content == "API response content"
                    assert not response.cache_hit
                    assert response.optimization_applied
                    
                    # Verify API was called with correct parameters
                    mock_api.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collaborate_timeout_handling(self):
        """Test timeout handling in collaboration."""
        messages = [{"role": "user", "content": "Timeout test"}]
        context = CollaborationContext(session_id="timeout_session")
        
        # Mock cache miss
        with patch.object(self.client.optimization_engine, 'apply_predictive_caching') as mock_cache:
            mock_cache.return_value = Mock(cache_hit=False, cache_key="timeout_key")
            
            # Mock routing
            with patch.object(self.client.optimization_engine, 'optimize_request_routing') as mock_routing:
                mock_routing.return_value = Mock(
                    target_endpoint="slow-endpoint",
                    predicted_latency=2.0,
                    fallback_options=["backup-endpoint"],
                    to_dict=lambda: {"endpoint": "slow-endpoint"}
                )
                
                # Mock API timeout
                with patch.object(self.client, '_make_claude_request', new_callable=AsyncMock) as mock_api:
                    mock_api.side_effect = asyncio.TimeoutError()
                    
                    with pytest.raises(TimeoutError):
                        await self.client.collaborate(messages, context)
    
    @pytest.mark.asyncio
    async def test_progressive_response_delivery(self):
        """Test progressive response delivery for high-latency scenarios."""
        messages = [{"role": "user", "content": "Progressive test"}]
        context = CollaborationContext(session_id="progressive_session", priority=2)
        
        # Mock cache miss
        with patch.object(self.client.optimization_engine, 'apply_predictive_caching') as mock_cache:
            mock_cache.return_value = Mock(cache_hit=False, cache_key="prog_key")
            
            # Mock high-latency routing decision
            with patch.object(self.client.optimization_engine, 'optimize_request_routing') as mock_routing:
                mock_routing.return_value = Mock(
                    target_endpoint="slow-endpoint",
                    predicted_latency=2.0,  # Above target latency
                    to_dict=lambda: {"endpoint": "slow-endpoint"}
                )
                
                # Mock API response (fast completion)
                mock_api_response = Mock()
                mock_api_response.content = "Progressive response"
                
                with patch.object(self.client, '_make_claude_request', new_callable=AsyncMock) as mock_api:
                    mock_api.return_value = mock_api_response
                    
                    response = await self.client.collaborate(messages, context)
                    
                    assert isinstance(response, OptimizedResponse)
                    assert response.content == "Progressive response"
    
    def test_bandwidth_metrics_generation(self):
        """Test bandwidth metrics generation."""
        # Add some interaction data
        session_id = "metrics_test"
        self.client.turn_counts[session_id] = 3
        
        # Mock bandwidth monitor with data
        if self.client.bandwidth_monitor:
            with patch.object(self.client.bandwidth_monitor, 'generate_performance_report') as mock_report:
                mock_report.return_value = Mock(
                    turns_per_second=0.15,
                    efficiency_ratio=1.2,
                    bandwidth_degradation=0.1
                )
                
                metrics = self.client._get_bandwidth_metrics(session_id)
                
                assert "turns_per_second" in metrics
                assert "efficiency_ratio" in metrics
                assert "bandwidth_degradation" in metrics
                assert metrics["turns_per_second"] == 0.15
    
    def test_performance_report_generation(self):
        """Test comprehensive performance report generation."""
        # Add some request data
        self.client.total_requests = 10
        self.client.optimized_requests = 8
        
        # Mock engine performance report
        with patch.object(self.client.optimization_engine, 'get_performance_report') as mock_engine_report:
            mock_engine_report.return_value = {
                "cache_performance": {"hit_rate": 0.6},
                "latency_performance": {"mean_latency": 0.4}
            }
            
            # Mock bandwidth monitor report
            if self.client.bandwidth_monitor:
                with patch.object(self.client.bandwidth_monitor, 'generate_performance_report') as mock_bandwidth_report:
                    mock_bandwidth_report.return_value = Mock(
                        to_dict=lambda: {"efficiency": 0.15, "degradation": 0.1}
                    )
                    
                    report = self.client.get_performance_report()
                    
                    assert "request_stats" in report
                    assert "engine_performance" in report
                    assert report["request_stats"]["total_requests"] == 10
                    assert report["request_stats"]["optimization_rate"] == 0.8
    
    def test_session_reset(self):
        """Test session reset functionality."""
        session_id = "reset_test"
        
        # Add session data
        self.client.turn_counts[session_id] = 5
        self.client.active_sessions[session_id] = CollaborationContext(session_id=session_id)
        
        # Reset session
        self.client.reset_session(session_id)
        
        # Verify cleanup
        assert session_id not in self.client.turn_counts
        assert session_id not in self.client.active_sessions
    
    def test_research_data_export(self):
        """Test research data export functionality."""
        # Mock component data exports
        with patch.object(self.client, 'get_performance_report') as mock_perf:
            mock_perf.return_value = {"performance": "data"}
            
            if self.client.bandwidth_monitor:
                with patch.object(self.client.bandwidth_monitor, 'export_data') as mock_bandwidth:
                    mock_bandwidth.return_value = {"bandwidth": "data"}
                    
                    if self.client.temporal_analyzer:
                        with patch.object(self.client.temporal_analyzer, 'export_temporal_analysis') as mock_temporal:
                            mock_temporal.return_value = {"temporal": "data"}
                            
                            export_data = self.client.export_research_data()
                            
                            assert "performance_report" in export_data
                            assert "optimization_config" in export_data
                            assert export_data["optimization_config"]["level"] == "balanced"
    
    def test_synchronous_compatibility_methods(self):
        """Test synchronous compatibility methods."""
        # Test messages_create compatibility
        with patch('asyncio.new_event_loop') as mock_loop_new:
            with patch('asyncio.set_event_loop') as mock_loop_set:
                mock_loop = Mock()
                mock_loop_new.return_value = mock_loop
                mock_loop.run_until_complete.return_value = OptimizedResponse(
                    content="Sync response",
                    response_time=0.3,
                    cache_hit=False,
                    optimization_applied=True,
                    bandwidth_metrics={}
                )
                
                response = self.client.messages_create(
                    messages=[{"role": "user", "content": "Sync test"}]
                )
                
                assert isinstance(response, OptimizedResponse)
                assert response.content == "Sync response"
                mock_loop.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_claude_request_formatting(self):
        """Test Claude API request formatting."""
        messages = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Test response"}
        ]
        
        # Mock Claude API response
        mock_response = Mock()
        mock_response.content = "Claude API response"
        
        # Mock the actual Claude client call
        async def mock_create(*args, **kwargs):
            return mock_response
        
        self.mock_claude_client.messages.create = mock_create
        
        # Mock asyncio.get_event_loop and run_in_executor
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop
            
            async def mock_run_in_executor(executor, func):
                return func()
            
            mock_loop.run_in_executor = mock_run_in_executor
            
            response = await self.client._make_claude_request(
                messages,
                model="claude-3-sonnet-20240229",
                max_tokens=100
            )
            
            assert response.content == "Claude API response"


class TestIntegrationScenarios:
    """Integration tests for realistic collaboration scenarios."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        # Mock anthropic module
        self.mock_anthropic = Mock()
        self.mock_claude_client = Mock()
        self.mock_anthropic.Client.return_value = self.mock_claude_client
        
        self.anthropic_patcher = patch('src.tboptimizer.integrations.claude_adapter.anthropic', self.mock_anthropic)
        self.anthropic_patcher.start()
        
        self.client = ClaudeOptimizedClient(
            api_key="integration_test_key",
            optimization_level="research",
            enable_monitoring=True
        )
    
    def teardown_method(self):
        """Cleanup integration test fixtures."""
        self.anthropic_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation_optimization(self):
        """Test optimization across multiple conversation turns."""
        session_id = "multi_turn_session"
        context = CollaborationContext(session_id=session_id, task_type="analysis")
        
        # Mock API responses for multiple turns
        turn_responses = [
            "First analysis response",
            "Second detailed response", 
            "Final summary response"
        ]
        
        for i, expected_content in enumerate(turn_responses):
            mock_response = Mock()
            mock_response.content = expected_content
            
            # Mock the collaboration flow
            with patch.object(self.client.optimization_engine, 'apply_predictive_caching') as mock_cache:
                mock_cache.return_value = Mock(
                    cache_hit=False,
                    cache_key=f"turn_{i}_key",
                    confidence=0.0
                )
                
                with patch.object(self.client.optimization_engine, 'optimize_request_routing') as mock_routing:
                    mock_routing.return_value = Mock(
                        target_endpoint="primary",
                        predicted_latency=0.3,
                        to_dict=lambda: {"endpoint": "primary"}
                    )
                    
                    with patch.object(self.client, '_make_claude_request', new_callable=AsyncMock) as mock_api:
                        mock_api.return_value = mock_response
                        
                        messages = [{"role": "user", "content": f"Turn {i+1} message"}]
                        response = await self.client.collaborate(messages, context)
                        
                        assert response.content == expected_content
                        assert session_id in self.client.turn_counts
                        assert self.client.turn_counts[session_id] == i + 1
    
    @pytest.mark.asyncio
    async def test_high_priority_fast_path(self):
        """Test fast path optimization for high-priority requests."""
        context = CollaborationContext(
            session_id="priority_session",
            priority=1,  # High priority
            task_type="urgent_analysis"
        )
        
        messages = [{"role": "user", "content": "Urgent analysis needed"}]
        
        # Mock cache miss but fast routing
        with patch.object(self.client.optimization_engine, 'apply_predictive_caching') as mock_cache:
            mock_cache.return_value = Mock(cache_hit=False, cache_key="urgent_key")
            
            with patch.object(self.client.optimization_engine, 'optimize_request_routing') as mock_routing:
                mock_routing.return_value = Mock(
                    target_endpoint="fast-endpoint",
                    predicted_latency=0.2,  # Very fast
                    to_dict=lambda: {"endpoint": "fast-endpoint", "priority": "high"}
                )
                
                mock_response = Mock()
                mock_response.content = "Urgent response"
                
                with patch.object(self.client, '_make_claude_request', new_callable=AsyncMock) as mock_api:
                    mock_api.return_value = mock_response
                    
                    start_time = time.time()
                    response = await self.client.collaborate(messages, context)
                    duration = time.time() - start_time
                    
                    assert response.content == "Urgent response"
                    # Should complete quickly (allowing for mock overhead)
                    assert duration < 1.0
    
    @pytest.mark.asyncio
    async def test_bandwidth_degradation_scenario(self):
        """Test bandwidth degradation detection and handling."""
        context = CollaborationContext(session_id="degradation_test")
        
        # Simulate multiple slow responses to trigger degradation
        messages = [{"role": "user", "content": "Slow response test"}]
        
        with patch.object(self.client.optimization_engine, 'apply_predictive_caching') as mock_cache:
            mock_cache.return_value = Mock(cache_hit=False, cache_key="slow_key")
            
            with patch.object(self.client.optimization_engine, 'optimize_request_routing') as mock_routing:
                mock_routing.return_value = Mock(
                    target_endpoint="slow-endpoint",
                    predicted_latency=3.0,  # Above degradation threshold
                    to_dict=lambda: {"endpoint": "slow-endpoint"}
                )
                
                # Mock slow API response
                async def slow_api_call(*args, **kwargs):
                    await asyncio.sleep(0.1)  # Simulate delay
                    mock_response = Mock()
                    mock_response.content = "Slow response"
                    return mock_response
                
                with patch.object(self.client, '_make_claude_request', new_callable=AsyncMock) as mock_api:
                    mock_api.side_effect = slow_api_call
                    
                    response = await self.client.collaborate(messages, context)
                    
                    # Should still get response but with degradation metrics
                    assert response.content == "Slow response"
                    
                    # Check if bandwidth monitor detected degradation
                    if self.client.bandwidth_monitor:
                        # Degradation should be detected for slow responses
                        assert self.client.bandwidth_monitor.circuit_breaker_triggered or response.response_time > 0.05


@pytest.fixture
def mock_client():
    """Fixture providing mocked client for testing."""
    with patch('src.tboptimizer.integrations.claude_adapter.anthropic') as mock_anthropic:
        mock_claude_client = Mock()
        mock_anthropic.Client.return_value = mock_claude_client
        
        client = ClaudeOptimizedClient(
            api_key="fixture_key",
            optimization_level="balanced"
        )
        
        yield client


class TestErrorHandling:
    """Test suite for error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_client):
        """Test handling of API errors."""
        messages = [{"role": "user", "content": "Error test"}]
        context = CollaborationContext(session_id="error_session")
        
        # Mock cache miss
        with patch.object(mock_client.optimization_engine, 'apply_predictive_caching') as mock_cache:
            mock_cache.return_value = Mock(cache_hit=False, cache_key="error_key")
            
            with patch.object(mock_client.optimization_engine, 'optimize_request_routing') as mock_routing:
                mock_routing.return_value = Mock(
                    target_endpoint="error-endpoint",
                    predicted_latency=0.3,
                    fallback_options=["backup-endpoint"],
                    to_dict=lambda: {"endpoint": "error-endpoint"}
                )
                
                # Mock API error
                with patch.object(mock_client, '_make_claude_request', new_callable=AsyncMock) as mock_api:
                    mock_api.side_effect = Exception("API Error")
                    
                    with pytest.raises(Exception) as exc_info:
                        await mock_client.collaborate(messages, context)
                    
                    assert "API Error" in str(exc_info.value)
    
    def test_invalid_optimization_level(self):
        """Test handling of invalid optimization level."""
        with patch('src.tboptimizer.integrations.claude_adapter.anthropic'):
            with pytest.raises(ValueError):
                ClaudeOptimizedClient(
                    api_key="test_key",
                    optimization_level="invalid_level"
                )


if __name__ == "__main__":
    pytest.main([__file__])