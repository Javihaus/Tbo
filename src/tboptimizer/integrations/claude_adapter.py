"""Claude API adapter with bandwidth optimization.

Drop-in replacement for Claude API with temporal bandwidth optimization
based on research findings. Maintains API compatibility while adding
sub-500ms response targeting and collaborative efficiency monitoring.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass
import json

try:
    import anthropic
except ImportError:
    anthropic = None

from ..core.bandwidth_monitor import BandwidthMonitor, CollaborationMetrics
from ..core.optimization_engine import OptimizationEngine, OptimizationLevel, RoutingDecision
from ..core.temporal_analyzer import TemporalAnalyzer


@dataclass
class CollaborationContext:
    """Context for collaborative AI interactions."""
    
    session_id: str
    user_id: Optional[str] = None
    task_type: str = "general"
    complexity: int = 1
    priority: int = 5
    conversation_history: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OptimizedResponse:
    """Enhanced response with optimization metrics."""
    
    content: str
    response_time: float
    cache_hit: bool
    optimization_applied: bool
    bandwidth_metrics: Dict[str, Any]
    routing_decision: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    
    # Claude API compatibility
    @property
    def text(self) -> str:
        """Compatibility with Claude response.text"""
        return self.content
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "response_time": self.response_time,
            "cache_hit": self.cache_hit,
            "optimization_applied": self.optimization_applied,
            "bandwidth_metrics": self.bandwidth_metrics,
            "routing_decision": self.routing_decision,
            "confidence": self.confidence
        }


class ClaudeOptimizedClient:
    """Drop-in replacement for Claude API with bandwidth optimization.
    
    Provides identical interface to anthropic.Client while adding:
    - Sub-500ms response targeting
    - Real-time bandwidth monitoring  
    - Predictive caching
    - Progressive response delivery
    - Temporal pattern analysis
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 optimization_level: Union[str, OptimizationLevel] = "balanced",
                 enable_monitoring: bool = True,
                 cache_size: int = 1000,
                 target_latency: float = 0.5):
        """Initialize optimized Claude client.
        
        Args:
            api_key: Anthropic API key (if None, uses environment variable)
            optimization_level: Optimization preset ("speed", "balanced", "quality", "research")
            enable_monitoring: Enable real-time bandwidth monitoring
            cache_size: Maximum cache entries
            target_latency: Target response latency in seconds
        """
        if anthropic is None:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        # Initialize Claude client
        self.claude_client = anthropic.Client(api_key=api_key)
        
        # Parse optimization level
        if isinstance(optimization_level, str):
            optimization_level = OptimizationLevel(optimization_level)
        
        # Initialize optimization components
        self.optimization_engine = OptimizationEngine(
            optimization_level=optimization_level,
            target_latency=target_latency,
            cache_size=cache_size
        )
        
        self.bandwidth_monitor = BandwidthMonitor(
            degradation_threshold=2.0,  # Circuit breaker at 2s per research
            baseline_efficiency=0.125   # Research baseline
        ) if enable_monitoring else None
        
        self.temporal_analyzer = TemporalAnalyzer() if enable_monitoring else None
        
        # Session tracking
        self.active_sessions: Dict[str, CollaborationContext] = {}
        self.turn_counts: Dict[str, int] = {}
        
        # Performance tracking
        self.total_requests = 0
        self.optimized_requests = 0
        
    async def collaborate(self, 
                         messages: List[Dict[str, str]],
                         context: CollaborationContext,
                         model: str = "claude-3-sonnet-20240229",
                         max_tokens: int = 1000,
                         **kwargs) -> OptimizedResponse:
        """Main collaborative interface with optimization.
        
        Args:
            messages: List of conversation messages
            context: Collaboration context and metadata
            model: Claude model to use
            max_tokens: Maximum response tokens
            **kwargs: Additional Claude API parameters
            
        Returns:
            OptimizedResponse with optimization metrics
        """
        start_time = time.time()
        self.total_requests += 1
        
        # Track session and turn count
        session_id = context.session_id
        if session_id not in self.turn_counts:
            self.turn_counts[session_id] = 0
        self.turn_counts[session_id] += 1
        
        # Apply predictive caching
        conversation_state = {
            "messages": messages,
            "context": context.metadata,
            "task_type": context.task_type,
            "session_id": session_id
        }
        
        cache_strategy = self.optimization_engine.apply_predictive_caching(conversation_state)
        
        # Check for cache hit
        if cache_strategy.cache_hit:
            cached_response = self.optimization_engine.response_cache[cache_strategy.cache_key]
            response_time = time.time() - start_time
            
            # Update monitoring
            if self.bandwidth_monitor:
                self.bandwidth_monitor.track_interaction(
                    response_time=response_time,
                    turn_count=self.turn_counts[session_id],
                    context_length=len(str(messages)),
                    task_complexity=context.complexity
                )
            
            return OptimizedResponse(
                content=cached_response["response"],
                response_time=response_time,
                cache_hit=True,
                optimization_applied=True,
                bandwidth_metrics=self._get_bandwidth_metrics(session_id),
                confidence=cache_strategy.confidence
            )
        
        # Apply request routing optimization
        available_endpoints = ["claude-api-primary", "claude-api-secondary"]  # Example endpoints
        routing_decision = self.optimization_engine.optimize_request_routing(
            context={
                "type": context.task_type,
                "complexity": context.complexity,
                "priority": context.priority
            },
            available_endpoints=available_endpoints
        )
        
        # Check if we should apply progressive delivery
        progressive_delivery = (
            routing_decision.predicted_latency > self.optimization_engine.target_latency
            and context.priority <= 3
        )
        
        if progressive_delivery:
            return await self._progressive_response_delivery(
                messages, context, model, max_tokens, start_time, cache_strategy, routing_decision, **kwargs
            )
        else:
            return await self._standard_response(
                messages, context, model, max_tokens, start_time, cache_strategy, routing_decision, **kwargs
            )
    
    async def _standard_response(self,
                                messages: List[Dict[str, str]],
                                context: CollaborationContext,
                                model: str,
                                max_tokens: int,
                                start_time: float,
                                cache_strategy,
                                routing_decision: RoutingDecision,
                                **kwargs) -> OptimizedResponse:
        """Standard optimized response generation."""
        
        # Apply timeout based on optimization level
        timeout = self.optimization_engine.config["timeout"]
        
        try:
            # Make Claude API call with timeout
            response = await asyncio.wait_for(
                self._make_claude_request(messages, model, max_tokens, **kwargs),
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            self.optimized_requests += 1
            
            # Cache the response
            cache_key = cache_strategy.cache_key
            original_text = " ".join([msg.get("content", "") for msg in messages])
            self.optimization_engine.cache_response(cache_key, response.content, original_text)
            
            # Update performance metrics
            self.optimization_engine.update_performance_metrics(
                endpoint=routing_decision.target_endpoint,
                actual_latency=response_time,
                success=True
            )
            
            # Update monitoring
            if self.bandwidth_monitor:
                metrics = self.bandwidth_monitor.track_interaction(
                    response_time=response_time,
                    turn_count=self.turn_counts[context.session_id],
                    context_length=len(str(messages)),
                    task_complexity=context.complexity
                )
                
                if self.temporal_analyzer:
                    self.temporal_analyzer.add_measurement(
                        timestamp=time.time(),
                        response_time=response_time,
                        efficiency=metrics["efficiency"]
                    )
            
            return OptimizedResponse(
                content=response.content,
                response_time=response_time,
                cache_hit=False,
                optimization_applied=True,
                bandwidth_metrics=self._get_bandwidth_metrics(context.session_id),
                routing_decision=routing_decision.to_dict(),
                confidence=1.0
            )
            
        except asyncio.TimeoutError:
            # Fallback to cached response or error
            response_time = time.time() - start_time
            
            # Update performance metrics for timeout
            self.optimization_engine.update_performance_metrics(
                endpoint=routing_decision.target_endpoint,
                actual_latency=response_time,
                success=False
            )
            
            # Try fallback endpoints
            for fallback_endpoint in routing_decision.fallback_options:
                try:
                    response = await asyncio.wait_for(
                        self._make_claude_request(messages, model, max_tokens, **kwargs),
                        timeout=timeout * 1.5  # Slightly longer timeout for fallback
                    )
                    
                    fallback_response_time = time.time() - start_time
                    
                    return OptimizedResponse(
                        content=response.content,
                        response_time=fallback_response_time,
                        cache_hit=False,
                        optimization_applied=True,
                        bandwidth_metrics=self._get_bandwidth_metrics(context.session_id),
                        routing_decision={"fallback_used": fallback_endpoint},
                        confidence=0.8
                    )
                    
                except asyncio.TimeoutError:
                    continue
            
            # All options exhausted
            raise TimeoutError(f"Request timed out after {response_time:.2f}s")
    
    async def _progressive_response_delivery(self,
                                           messages: List[Dict[str, str]],
                                           context: CollaborationContext,
                                           model: str,
                                           max_tokens: int,
                                           start_time: float,
                                           cache_strategy,
                                           routing_decision: RoutingDecision,
                                           **kwargs) -> OptimizedResponse:
        """Progressive response delivery for high-latency scenarios."""
        
        # Immediate acknowledgment
        acknowledgment = "Processing your request..."
        ack_time = time.time() - start_time
        
        # Start background processing
        response_task = asyncio.create_task(
            self._make_claude_request(messages, model, max_tokens, **kwargs)
        )
        
        # Wait for either completion or acknowledgment timeout
        try:
            response = await asyncio.wait_for(response_task, timeout=0.2)  # 200ms for immediate response
            response_time = time.time() - start_time
            
            return OptimizedResponse(
                content=response.content,
                response_time=response_time,
                cache_hit=False,
                optimization_applied=True,
                bandwidth_metrics=self._get_bandwidth_metrics(context.session_id),
                routing_decision=routing_decision.to_dict(),
                confidence=1.0
            )
            
        except asyncio.TimeoutError:
            # Return progressive response
            try:
                response = await response_task  # Wait for full completion
                full_response_time = time.time() - start_time
                
                return OptimizedResponse(
                    content=response.content,
                    response_time=full_response_time,
                    cache_hit=False,
                    optimization_applied=True,
                    bandwidth_metrics=self._get_bandwidth_metrics(context.session_id),
                    routing_decision=routing_decision.to_dict(),
                    confidence=1.0
                )
                
            except Exception as e:
                # Return acknowledgment if full response fails
                return OptimizedResponse(
                    content=acknowledgment,
                    response_time=ack_time,
                    cache_hit=False,
                    optimization_applied=True,
                    bandwidth_metrics=self._get_bandwidth_metrics(context.session_id),
                    routing_decision={"error": str(e)},
                    confidence=0.1
                )
    
    async def _make_claude_request(self,
                                  messages: List[Dict[str, str]],
                                  model: str,
                                  max_tokens: int,
                                  **kwargs):
        """Make actual Claude API request."""
        # Convert messages to Claude format if needed
        claude_messages = []
        for msg in messages:
            claude_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Make synchronous call in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.claude_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=claude_messages,
                **kwargs
            )
        )
        
        return response
    
    def _get_bandwidth_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get current bandwidth metrics for session."""
        if not self.bandwidth_monitor:
            return {}
        
        metrics = self.bandwidth_monitor.generate_performance_report()
        
        base_metrics = {
            "turns_per_second": metrics.turns_per_second,
            "efficiency_ratio": metrics.efficiency_ratio,
            "bandwidth_degradation": metrics.bandwidth_degradation,
            "circuit_breaker_triggered": self.bandwidth_monitor.circuit_breaker_triggered
        }
        
        # Add temporal analysis if available
        if self.temporal_analyzer:
            rhythm_analysis = self.temporal_analyzer.detect_collaborative_rhythm()
            base_metrics.update({
                "natural_pace": rhythm_analysis.natural_pace,
                "rhythm_strength": rhythm_analysis.rhythm_strength,
                "synchronization_score": rhythm_analysis.synchronization_score
            })
        
        return base_metrics
    
    # Claude API compatibility methods
    def messages_create(self, **kwargs) -> OptimizedResponse:
        """Synchronous compatibility method."""
        # Convert to async call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create basic context for compatibility
            messages = kwargs.get("messages", [])
            context = CollaborationContext(
                session_id=f"sync_{int(time.time())}",
                task_type="general"
            )
            
            result = loop.run_until_complete(
                self.collaborate(messages, context, **kwargs)
            )
            return result
        finally:
            loop.close()
    
    async def messages_create_async(self, **kwargs) -> OptimizedResponse:
        """Async compatibility method."""
        messages = kwargs.get("messages", [])
        context = CollaborationContext(
            session_id=f"async_{int(time.time())}",
            task_type="general"
        )
        
        return await self.collaborate(messages, context, **kwargs)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        engine_report = self.optimization_engine.get_performance_report()
        
        report = {
            "request_stats": {
                "total_requests": self.total_requests,
                "optimized_requests": self.optimized_requests,
                "optimization_rate": self.optimized_requests / self.total_requests if self.total_requests > 0 else 0
            },
            "engine_performance": engine_report
        }
        
        if self.bandwidth_monitor:
            bandwidth_metrics = self.bandwidth_monitor.generate_performance_report()
            report["bandwidth_metrics"] = bandwidth_metrics.to_dict()
        
        if self.temporal_analyzer:
            temporal_analysis = self.temporal_analyzer.export_temporal_analysis()
            report["temporal_analysis"] = temporal_analysis
        
        return report
    
    def reset_session(self, session_id: str):
        """Reset monitoring for a specific session."""
        if session_id in self.turn_counts:
            del self.turn_counts[session_id]
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        if self.bandwidth_monitor:
            self.bandwidth_monitor.reset_session()
    
    def export_research_data(self, format: str = "dict") -> Any:
        """Export data for research analysis."""
        data = {
            "performance_report": self.get_performance_report(),
            "optimization_config": {
                "level": self.optimization_engine.optimization_level.value,
                "target_latency": self.optimization_engine.target_latency
            }
        }
        
        if self.bandwidth_monitor:
            data["bandwidth_data"] = self.bandwidth_monitor.export_data(format)
        
        if self.temporal_analyzer:
            data["temporal_data"] = self.temporal_analyzer.export_temporal_analysis()
        
        return data