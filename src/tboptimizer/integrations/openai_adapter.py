"""OpenAI API adapter with bandwidth optimization.

Provides bandwidth optimization for OpenAI API calls with the same
optimization strategies as the Claude adapter.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union

try:
    import openai
except ImportError:
    openai = None

from .claude_adapter import CollaborationContext, OptimizedResponse
from ..core.bandwidth_monitor import BandwidthMonitor
from ..core.optimization_engine import OptimizationEngine, OptimizationLevel
from ..core.temporal_analyzer import TemporalAnalyzer


class OpenAIOptimizedClient:
    """OpenAI client with bandwidth optimization."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 optimization_level: Union[str, OptimizationLevel] = "balanced",
                 enable_monitoring: bool = True,
                 cache_size: int = 1000,
                 target_latency: float = 0.5):
        """Initialize optimized OpenAI client."""
        if openai is None:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.openai_client = openai.AsyncOpenAI(api_key=api_key)
        
        if isinstance(optimization_level, str):
            optimization_level = OptimizationLevel(optimization_level)
        
        self.optimization_engine = OptimizationEngine(
            optimization_level=optimization_level,
            target_latency=target_latency,
            cache_size=cache_size
        )
        
        self.bandwidth_monitor = BandwidthMonitor() if enable_monitoring else None
        self.temporal_analyzer = TemporalAnalyzer() if enable_monitoring else None
        
        self.turn_counts: Dict[str, int] = {}
        self.total_requests = 0
        self.optimized_requests = 0
    
    async def collaborate(self,
                         messages: List[Dict[str, str]], 
                         context: CollaborationContext,
                         model: str = "gpt-3.5-turbo",
                         **kwargs) -> OptimizedResponse:
        """Collaborative interface with optimization."""
        start_time = time.time()
        self.total_requests += 1
        
        session_id = context.session_id
        if session_id not in self.turn_counts:
            self.turn_counts[session_id] = 0
        self.turn_counts[session_id] += 1
        
        # Apply caching
        conversation_state = {
            "messages": messages,
            "context": context.metadata,
            "task_type": context.task_type,
            "session_id": session_id
        }
        
        cache_strategy = self.optimization_engine.apply_predictive_caching(conversation_state)
        
        if cache_strategy.cache_hit:
            cached_response = self.optimization_engine.response_cache[cache_strategy.cache_key]
            response_time = time.time() - start_time
            
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
        
        # Make OpenAI API call
        try:
            timeout = self.optimization_engine.config["timeout"]
            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                ),
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            self.optimized_requests += 1
            
            content = response.choices[0].message.content
            
            # Cache response
            cache_key = cache_strategy.cache_key
            original_text = " ".join([msg.get("content", "") for msg in messages])
            self.optimization_engine.cache_response(cache_key, content, original_text)
            
            # Update monitoring
            if self.bandwidth_monitor:
                metrics = self.bandwidth_monitor.track_interaction(
                    response_time=response_time,
                    turn_count=self.turn_counts[session_id],
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
                content=content,
                response_time=response_time,
                cache_hit=False,
                optimization_applied=True,
                bandwidth_metrics=self._get_bandwidth_metrics(session_id),
                confidence=1.0
            )
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            raise TimeoutError(f"Request timed out after {response_time:.2f}s")
    
    def _get_bandwidth_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get bandwidth metrics."""
        if not self.bandwidth_monitor:
            return {}
        
        metrics = self.bandwidth_monitor.generate_performance_report()
        return {
            "turns_per_second": metrics.turns_per_second,
            "efficiency_ratio": metrics.efficiency_ratio,
            "bandwidth_degradation": metrics.bandwidth_degradation
        }