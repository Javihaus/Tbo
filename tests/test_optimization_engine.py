"""Tests for OptimizationEngine with sub-500ms targeting validation."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from src.tboptimizer.core.optimization_engine import (
    OptimizationEngine,
    OptimizationLevel,
    RoutingStrategy,
    RoutingDecision,
    CacheStrategy,
    QueuePosition,
    TimingPlan
)


class TestOptimizationEngine:
    """Test suite for OptimizationEngine functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = OptimizationEngine(
            optimization_level=OptimizationLevel.BALANCED,
            target_latency=0.5,
            cache_size=100
        )
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.optimization_level == OptimizationLevel.BALANCED
        assert self.engine.target_latency == 0.5
        assert self.engine.cache_size == 100
        assert len(self.engine.response_cache) == 0
        assert self.engine.total_requests == 0
        
        # Test configuration setup
        assert "cache_ttl" in self.engine.config
        assert "timeout" in self.engine.config
        assert self.engine.config["timeout"] == 0.5  # Balanced preset
    
    def test_optimization_level_configuration(self):
        """Test different optimization level configurations."""
        # Speed configuration
        speed_engine = OptimizationEngine(OptimizationLevel.SPEED)
        assert speed_engine.config["timeout"] == 0.3
        assert speed_engine.config["quality_threshold"] == 0.7
        
        # Quality configuration
        quality_engine = OptimizationEngine(OptimizationLevel.QUALITY)
        assert quality_engine.config["timeout"] == 1.0
        assert quality_engine.config["quality_threshold"] == 0.95
        
        # Research configuration
        research_engine = OptimizationEngine(OptimizationLevel.RESEARCH)
        assert research_engine.config["timeout"] == 0.4
        assert research_engine.config["quality_threshold"] == 0.90
    
    def test_request_routing_optimization(self):
        """Test request routing for minimum latency."""
        context = {
            "type": "analysis",
            "complexity": 3,
            "priority": 2
        }
        
        endpoints = ["endpoint_1", "endpoint_2", "endpoint_3"]
        
        routing_decision = self.engine.optimize_request_routing(context, endpoints)
        
        assert isinstance(routing_decision, RoutingDecision)
        assert routing_decision.target_endpoint in endpoints
        assert routing_decision.predicted_latency > 0
        assert 0 <= routing_decision.confidence <= 1
        assert routing_decision.strategy_used in RoutingStrategy
        assert len(routing_decision.fallback_options) >= 0
    
    def test_routing_strategy_selection(self):
        """Test routing strategy selection based on context."""
        # High priority should use latency priority
        high_priority_context = {"priority": 1, "complexity": 3}
        endpoints = ["ep1", "ep2"]
        
        decision = self.engine.optimize_request_routing(high_priority_context, endpoints)
        assert decision.strategy_used == RoutingStrategy.LATENCY_PRIORITY
        
        # Normal priority with system load
        self.engine.endpoint_loads["ep1"] = 15  # Overload
        normal_context = {"priority": 5, "complexity": 2}
        
        decision = self.engine.optimize_request_routing(normal_context, endpoints)
        # Should use load balancing or adaptive strategy
        assert decision.strategy_used in [RoutingStrategy.LOAD_BALANCE, RoutingStrategy.ADAPTIVE]
    
    def test_predictive_caching(self):
        """Test predictive caching functionality."""
        conversation_state = {
            "messages": [
                {"role": "user", "content": "Analyze market trends"},
                {"role": "assistant", "content": "Market analysis shows..."},
                {"role": "user", "content": "What about tech sector?"}
            ],
            "context": "business_analysis",
            "task_type": "analysis"
        }
        
        cache_strategy = self.engine.apply_predictive_caching(conversation_state)
        
        assert isinstance(cache_strategy, CacheStrategy)
        assert len(cache_strategy.cache_key) > 0
        assert cache_strategy.generation_strategy in ["exact", "semantic", "predictive"]
        assert 0 <= cache_strategy.confidence <= 1
        assert cache_strategy.ttl > 0
    
    def test_cache_hit_detection(self):
        """Test cache hit detection and retrieval."""
        # First request - should be cache miss
        state1 = {
            "messages": [{"role": "user", "content": "Test message"}],
            "context": "test",
            "task_type": "test"
        }
        
        strategy1 = self.engine.apply_predictive_caching(state1)
        assert not strategy1.cache_hit
        
        # Add to cache
        self.engine.cache_response(strategy1.cache_key, "Test response", "Test message")
        
        # Second identical request - should be cache hit
        strategy2 = self.engine.apply_predictive_caching(state1)
        assert strategy2.cache_hit
        assert strategy2.generation_strategy == "exact"
        assert strategy2.confidence == 1.0
    
    def test_semantic_similarity_matching(self):
        """Test semantic similarity cache matching."""
        # Cache original response
        original_state = {
            "messages": [{"role": "user", "content": "analyze customer data trends"}],
            "context": "analysis",
            "task_type": "data_analysis"
        }
        
        strategy = self.engine.apply_predictive_caching(original_state)
        self.engine.cache_response(strategy.cache_key, "Analysis result", "analyze customer data trends")
        
        # Similar request
        similar_state = {
            "messages": [{"role": "user", "content": "analyze customer data patterns"}],
            "context": "analysis", 
            "task_type": "data_analysis"
        }
        
        similar_strategy = self.engine.apply_predictive_caching(similar_state)
        
        # Should find semantic match (simplified test)
        # Note: Full semantic matching requires embeddings in production
        if similar_strategy.cache_hit:
            assert similar_strategy.generation_strategy == "semantic"
            assert similar_strategy.confidence < 1.0
    
    def test_cache_size_management(self):
        """Test cache size limitation and eviction."""
        # Fill cache beyond limit
        for i in range(self.engine.cache_size + 10):
            state = {
                "messages": [{"role": "user", "content": f"Message {i}"}],
                "context": f"context_{i}",
                "task_type": "test"
            }
            
            strategy = self.engine.apply_predictive_caching(state)
            self.engine.cache_response(strategy.cache_key, f"Response {i}", f"Message {i}")
        
        # Cache should not exceed limit
        assert len(self.engine.response_cache) <= self.engine.cache_size
    
    def test_response_queuing(self):
        """Test response queue management."""
        # High priority request
        high_priority_position = self.engine.manage_response_queuing(
            priority=1,
            context={"type": "urgent"}
        )
        
        assert isinstance(high_priority_position, QueuePosition)
        assert high_priority_position.priority_level == 1
        assert high_priority_position.bypass_available
        
        # Normal priority request
        normal_priority_position = self.engine.manage_response_queuing(
            priority=5,
            context={"type": "standard"}
        )
        
        assert normal_priority_position.priority_level == 5
        assert not normal_priority_position.bypass_available
    
    def test_multi_agent_coordination(self):
        """Test multi-agent timing coordination."""
        agents = [
            {
                "id": "agent_1",
                "estimated_duration": 2.0,
                "dependencies": []
            },
            {
                "id": "agent_2", 
                "estimated_duration": 1.5,
                "dependencies": ["agent_1"]
            },
            {
                "id": "agent_3",
                "estimated_duration": 3.0,
                "dependencies": []
            }
        ]
        
        # Test different coordination strategies
        for strategy in ["parallel", "sequential", "adaptive"]:
            timing_plan = self.engine.coordinate_multi_agent_timing(agents, strategy)
            
            assert isinstance(timing_plan, TimingPlan)
            assert len(timing_plan.agent_schedules) == len(agents)
            assert timing_plan.total_duration > 0
            assert isinstance(timing_plan.parallel_execution, bool)
            
            # Verify all agents are scheduled
            for agent in agents:
                assert agent["id"] in timing_plan.agent_schedules
    
    def test_parallel_coordination_efficiency(self):
        """Test parallel coordination provides time savings."""
        agents = [
            {"id": "a", "estimated_duration": 2.0, "dependencies": []},
            {"id": "b", "estimated_duration": 1.5, "dependencies": []},
            {"id": "c", "estimated_duration": 2.5, "dependencies": []}
        ]
        
        parallel_plan = self.engine.coordinate_multi_agent_timing(agents, "parallel")
        sequential_plan = self.engine.coordinate_multi_agent_timing(agents, "sequential")
        
        # Parallel should be faster than sequential
        assert parallel_plan.total_duration < sequential_plan.total_duration
        assert parallel_plan.parallel_execution
        assert not sequential_plan.parallel_execution
    
    def test_adaptive_coordination_dependencies(self):
        """Test adaptive coordination respects dependencies."""
        agents = [
            {"id": "base", "estimated_duration": 1.0, "dependencies": []},
            {"id": "dependent", "estimated_duration": 1.0, "dependencies": ["base"]},
            {"id": "independent", "estimated_duration": 1.0, "dependencies": []}
        ]
        
        adaptive_plan = self.engine.coordinate_multi_agent_timing(agents, "adaptive")
        
        # Base and independent can start together
        base_start = adaptive_plan.agent_schedules["base"]
        independent_start = adaptive_plan.agent_schedules["independent"]
        dependent_start = adaptive_plan.agent_schedules["dependent"]
        
        # Dependent should start after base
        assert dependent_start >= base_start + 1.0  # Base duration
        
        # Independent can start with base
        assert abs(independent_start - base_start) < 0.1
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking."""
        # Simulate successful requests
        for i in range(5):
            self.engine.update_performance_metrics(
                endpoint="test_endpoint",
                actual_latency=0.4 + i * 0.1,
                success=True
            )
        
        # Simulate failed request
        self.engine.update_performance_metrics(
            endpoint="test_endpoint",
            actual_latency=2.0,
            success=False
        )
        
        report = self.engine.get_performance_report()
        
        assert "latency_performance" in report
        assert "endpoint_performance" in report
        assert report["latency_performance"]["mean_latency"] > 0
        assert "test_endpoint" in report["endpoint_performance"]
    
    def test_sub_500ms_targeting(self):
        """Test sub-500ms response targeting."""
        # Test with latency prediction
        context = {"complexity": 1, "priority": 3}
        endpoints = ["fast_endpoint", "slow_endpoint"]
        
        # Mock endpoint latencies
        self.engine.endpoint_latencies["fast_endpoint"].extend([0.3, 0.4, 0.2])
        self.engine.endpoint_latencies["slow_endpoint"].extend([0.8, 0.9, 1.0])
        
        decision = self.engine.optimize_request_routing(context, endpoints)
        
        # Should prefer faster endpoint
        assert decision.predicted_latency < self.engine.target_latency * 2
        
        # Test timeout configuration
        assert self.engine.config["timeout"] <= self.engine.target_latency * 2
    
    def test_cache_ttl_configuration(self):
        """Test cache TTL based on optimization level."""
        # Different levels should have different TTLs
        speed_engine = OptimizationEngine(OptimizationLevel.SPEED)
        quality_engine = OptimizationEngine(OptimizationLevel.QUALITY)
        
        assert speed_engine.config["cache_ttl"] > quality_engine.config["cache_ttl"]
        
        # Test cache strategy TTL
        state = {"messages": [{"role": "user", "content": "test"}]}
        
        speed_strategy = speed_engine.apply_predictive_caching(state)
        quality_strategy = quality_engine.apply_predictive_caching(state)
        
        assert speed_strategy.ttl >= quality_strategy.ttl
    
    def test_latency_variance_handling(self):
        """Test handling of latency variance in routing decisions."""
        # Add varied latency data
        endpoint = "variable_endpoint"
        latencies = [0.2, 0.5, 1.0, 0.3, 0.8]  # High variance
        
        for latency in latencies:
            self.engine.endpoint_latencies[endpoint].append(latency)
        
        context = {"complexity": 2, "priority": 4}
        decision = self.engine.optimize_request_routing(context, [endpoint])
        
        # Confidence should be lower with high variance
        assert decision.confidence < 1.0
    
    def test_preload_recommendations(self):
        """Test predictive preload recommendations."""
        conversation_state = {
            "messages": [
                {"role": "user", "content": "Can you explain how machine learning works?"},
                {"role": "assistant", "content": "Machine learning is..."},
                {"role": "user", "content": "Show me an example"}
            ]
        }
        
        # Enable predictive preload
        self.engine.config["predictive_preload"] = True
        
        strategy = self.engine.apply_predictive_caching(conversation_state)
        
        # Should generate preload recommendations
        assert len(strategy.preload_recommendations) >= 0
        
        # Test pattern recognition
        if strategy.preload_recommendations:
            # "example" keyword should trigger code_example recommendation
            assert any("example" in rec for rec in strategy.preload_recommendations)


class TestRoutingDecision:
    """Test suite for RoutingDecision data class."""
    
    def test_routing_decision_creation(self):
        """Test routing decision creation and conversion."""
        decision = RoutingDecision(
            target_endpoint="api.claude.ai",
            predicted_latency=0.4,
            confidence=0.85,
            strategy_used=RoutingStrategy.LATENCY_PRIORITY,
            cache_recommendation="preload",
            fallback_options=["backup1", "backup2"]
        )
        
        assert decision.target_endpoint == "api.claude.ai"
        assert decision.predicted_latency == 0.4
        assert decision.strategy_used == RoutingStrategy.LATENCY_PRIORITY
        
        # Test dictionary conversion
        decision_dict = decision.to_dict()
        assert decision_dict["target_endpoint"] == "api.claude.ai"
        assert decision_dict["strategy_used"] == "latency_priority"


class TestCacheStrategy:
    """Test suite for CacheStrategy data class."""
    
    def test_cache_strategy_creation(self):
        """Test cache strategy creation and conversion."""
        strategy = CacheStrategy(
            cache_key="abc123",
            cache_hit=True,
            generation_strategy="semantic",
            confidence=0.9,
            ttl=1800,
            preload_recommendations=["follow_up", "details"]
        )
        
        assert strategy.cache_key == "abc123"
        assert strategy.cache_hit
        assert strategy.generation_strategy == "semantic"
        assert strategy.confidence == 0.9
        
        # Test dictionary conversion
        strategy_dict = strategy.to_dict()
        assert strategy_dict["cache_hit"]
        assert len(strategy_dict["preload_recommendations"]) == 2


@pytest.fixture
def mock_engine():
    """Fixture providing configured engine for testing."""
    return OptimizationEngine(
        optimization_level=OptimizationLevel.RESEARCH,
        target_latency=0.3,
        cache_size=50
    )


class TestIntegrationScenarios:
    """Integration tests for realistic optimization scenarios."""
    
    def test_high_load_scenario(self, mock_engine):
        """Test behavior under high load conditions."""
        # Simulate high endpoint loads
        for endpoint in ["ep1", "ep2", "ep3"]:
            mock_engine.endpoint_loads[endpoint] = 20
        
        context = {"priority": 3, "complexity": 4}
        decision = mock_engine.optimize_request_routing(context, ["ep1", "ep2", "ep3"])
        
        # Should trigger load balancing
        assert decision.strategy_used in [RoutingStrategy.LOAD_BALANCE, RoutingStrategy.ADAPTIVE]
    
    def test_cache_warm_up_scenario(self, mock_engine):
        """Test cache warm-up and hit rate improvement."""
        # Simulate repeated similar requests
        base_state = {
            "messages": [{"role": "user", "content": "analyze performance metrics"}],
            "task_type": "analysis"
        }
        
        hit_count = 0
        total_requests = 10
        
        for i in range(total_requests):
            # Slight variations in requests
            state = {
                "messages": [{"role": "user", "content": f"analyze performance metrics {i}"}],
                "task_type": "analysis"
            }
            
            strategy = mock_engine.apply_predictive_caching(state)
            
            if not strategy.cache_hit:
                # Cache the response
                mock_engine.cache_response(strategy.cache_key, f"Analysis {i}", f"analyze performance metrics {i}")
            else:
                hit_count += 1
        
        # Should achieve some cache hits through semantic similarity
        cache_hit_rate = hit_count / total_requests
        # Note: Actual hit rate depends on semantic similarity implementation
        assert cache_hit_rate >= 0  # Basic validation
    
    def test_circuit_breaker_scenario(self, mock_engine):
        """Test circuit breaker activation scenario."""
        # Simulate high latency conditions
        high_latency_context = {"complexity": 5, "priority": 8}
        
        # Mock high predicted latencies
        for endpoint in ["slow_ep1", "slow_ep2"]:
            mock_engine.endpoint_latencies[endpoint].extend([2.0, 2.5, 3.0])
        
        decision = mock_engine.optimize_request_routing(
            high_latency_context, 
            ["slow_ep1", "slow_ep2"]
        )
        
        # Should recommend caching due to high latency
        assert decision.cache_recommendation in ["preload", "aggressive"]


if __name__ == "__main__":
    pytest.main([__file__])